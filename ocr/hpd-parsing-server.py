# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=4.0.0",
#     "huggingface-hub",
#     "pillow",
#     "requests",
# ]
# ///

"""
Parse document images with HPD-Parsing via an in-job vLLM server.

HPD-Parsing (1B, InternVL3.5-1B vision + Qwen3-0.6B) parses a page with
*hierarchical parallel decoding*: a main layout branch emits `<BLOCK> <type>
[bbox]` headers and forks a child branch per region (children reuse the parent's
prefix KV), with a P-MTP medusa head drafting tokens inside each branch. It
scores 94.91 on OmniDocBench v1.6 at 4,752 TPS — 2.62x the fastest prior parser.

Accuracy is *below* the 0.9B leaders here (ovis-ocr2.py 96.58, paddleocr-vl-1.6.py
96.33). Reach for this one when you want throughput, or when you want the layout:
its native output carries a type + bbox + reading order per block, which this
recipe keeps in a structured `hpd_blocks` column alongside the markdown.

SERVER-ONLY BY DESIGN (no offline sibling recipe). Forking is a *scheduler*
feature of the customized vLLM build, so an offline `llm.generate()` batch loop
forfeits the one thing this model sells. The vendor agrees: the Docker image's
own entrypoint is a `vllm serve` command (the card's offline snippet is the
secondary path).

Requires the vendor image — the model needs a **fork of vLLM**
(`vllm-0.17.1+hpdparsing`, dynamic request forking + medusa P-MTP) that exists in
no upstream wheel. The image ships it in a python3.10 venv on PATH, plus CUDA
12.8 + FlashInfer + `MAX_PATCHES_WITH_RESIZE=true`.

NOTE the job command shape: unlike the `vllm/vllm-openai` images, this one has no
`uv`, so `hf jobs uv run --image ...` fails with `"uv": executable file not found`.
Bootstrap uv in the command instead (`hf jobs run`, image as a positional arg):

  hf jobs run --detach --flavor l4x1 -s HF_TOKEN --timeout 2h \\
      ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/hpd-parsing-vllm:latest-nvidia-gpu \\
      -- bash -lc 'pip install -q uv && uv run \\
          https://huggingface.co/datasets/uv-scripts/ocr/raw/main/hpd-parsing-server.py \\
          <input-dataset> <output-dataset>'

The driver has no torch/vllm deps, so `uv run` starts in seconds while the server
warms up. It spawns `vllm serve` itself when no server is reachable; pass
--server URL to drive an already-running or remote endpoint instead. To loosen
the image pin: drop it when the `<FORK>` scheduling lands in an upstream vLLM
release (or when Paddle publishes the fork to PyPI). Budget for a ~11 GB image
pull on a cold node (~7 min in testing) before anything runs.

Serve-flag provenance: build_serve_args() below is the vendor's own entrypoint
(`/home/hpd/entrypoint.sh` in the image), copied as-is except for the speculative
head — the entrypoint's `PaddlePaddle/HPD-Parsing/P-MTP` is neither a repo id nor
a directory, so vLLM rejects it outside the image's offline mode; we materialize
that subfolder locally instead (see resolve_spec_model). Two other flags are
deliberate and easy to "fix" by mistake:
- `--enable-prefix-caching` contradicts the recurring official OCR-serving pattern
  (SERVING.md: OCR never reuses images, so caches only cost memory). Here the
  forked children *share the parent branch's prefix KV*, so prefix caching is
  load-bearing, not waste. Leave it on.
- `--attention-config '{"use_prefill_query_quantization":true}'` is a fork-only
  flag with no upstream vLLM equivalent.

Output columns: `markdown` + `hpd_blocks` (JSON: type/bbox/text per block).
BBOX SPACE: 0-1000 **normalized**, not pixels (unlike surya-ocr.py's
`surya_blocks`). Convert with `px = value / 1000 * width`.

Model: PaddlePaddle/HPD-Parsing (Apache-2.0), paper arXiv:2607.18839
Block parsing + formula cleanup below are adapted from the model repo's
Apache-2.0 `eval/hpd_to_markdown.py` via the bbox-preserving `parse_blocks` in
multimodalart's demo Space (hugging-apps/hpd-parsing, `hpd_postprocess.py`).
"""

import argparse
import atexit
import base64
import concurrent.futures
import io
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import requests
from datasets import load_dataset
from huggingface_hub import DatasetCard, login, snapshot_download
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "PaddlePaddle/HPD-Parsing"
# The vendor entrypoint serves under this name; it is what the API `model` field wants.
SERVED_MODEL_NAME = "HPD-Parsing"
IMAGE_TAG = (
    "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/hpd-parsing-vllm:latest-nvidia-gpu"
)

# Prompts are card-verbatim. "with fork" enables hierarchical parallel decoding;
# the plain form is the standard single-trajectory parse. Don't "improve" either.
FORK_PROMPT = "document parsing with fork."
PLAIN_PROMPT = "document parsing."

DEFAULT_PORT = 8118  # the vendor entrypoint's default
DEFAULT_MAX_MODEL_LEN = 16384  # card value; the LM itself allows 40960
DEFAULT_MAX_TOKENS = 8000  # card value

def build_serve_args(port: int, max_model_len: int, spec_model: str) -> List[str]:
    """The serve command this script spawns when no server is reachable.

    Copied from the image's own /home/hpd/entrypoint.sh (see the docstring for the
    two flags that look wrong but aren't), with one required fix: `spec_model`.
    """
    return [
        "vllm", "serve", MODEL,
        "--trust-remote-code",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--served-model-name", SERVED_MODEL_NAME,
        "--max-model-len", str(max_model_len),
        "--limit-mm-per-prompt", '{"image": 1}',
        "--gpu-memory-utilization", "0.9",
        "--attention-backend", "FLASHINFER",
        "--attention-config", '{"use_prefill_query_quantization":true}',
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
        "--speculative-config",
        json.dumps(
            {"method": "medusa", "model": spec_model, "num_speculative_tokens": 6}
        ),
    ]


def resolve_spec_model() -> str:
    """Materialize the P-MTP speculative head locally and return its directory.

    The vendor entrypoint passes the speculative model as `PaddlePaddle/HPD-Parsing/P-MTP`,
    which vLLM rejects — it accepts a repo id or a local directory, and that string is
    neither ("Repo id must be in the form 'repo_name' or 'namespace/repo_name'"). It only
    works in the image's *offline* mode, where the same path happens to be a real
    directory. So pull just that subfolder (~644 MB) into the normal HF cache — the same
    cache vLLM then downloads the main weights into, so nothing is fetched twice.
    """
    logger.info(f"Fetching the P-MTP speculative head from {MODEL}")
    snapshot = snapshot_download(MODEL, allow_patterns=["P-MTP/*"])
    spec_dir = os.path.join(snapshot, "P-MTP")
    if not os.path.isfile(os.path.join(spec_dir, "config.json")):
        logger.error(f"P-MTP head missing a config.json under {spec_dir}")
        sys.exit(1)
    logger.info(f"P-MTP head at {spec_dir}")
    return spec_dir

# `hf jobs uv run --image` does NOT work here: the vendor image ships no uv, so the
# job dies with `"uv": executable file not found in $PATH`. Bootstrap it instead.
RUN_COMMAND = (
    "hf jobs run --detach --flavor l4x1 -s HF_TOKEN --timeout 2h \\\n"
    f"    {IMAGE_TAG} \\\n"
    "    -- bash -lc 'pip install -q uv && uv run \\\n"
    "        https://huggingface.co/datasets/uv-scripts/ocr/raw/main/hpd-parsing-server.py \\\n"
    "        <input-dataset> <output-dataset>'"
)


def ensure_output_columns_free(dataset, columns, overwrite=False):
    """Fail fast if an output column would collide with an existing input column.

    Adding a column that already exists silently overwrites it (e.g. a ground-truth
    `text`/`markdown` column) or crashes on push with a duplicate-column error only
    *after* inference has run. Catch it up front. With overwrite=True, drop the clashing
    column(s) here instead (logged) so the later add_column is clean.
    """
    clash = [c for c in columns if c in dataset.column_names]
    if not clash:
        return dataset
    if overwrite:
        logger.warning(f"--overwrite: replacing existing column(s) {clash}")
        return dataset.remove_columns(clash)
    logger.error(
        f"Output column(s) {clash} already exist in the input dataset "
        f"(columns: {dataset.column_names})."
    )
    logger.error("Choose a different --output-column, or pass --overwrite to replace them.")
    sys.exit(1)


def to_pil_image(image: Union[Image.Image, Dict[str, Any], str]) -> Image.Image:
    """Convert a dataset image cell (PIL image, bytes dict, or path) to RGB PIL."""
    if isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, dict) and "bytes" in image:
        pil_img = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, str):
        pil_img = Image.open(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    return pil_img.convert("RGB")


def encode_image(image, max_pixels: int) -> str:
    """RGB-convert, optionally downscale, return base64 JPEG.

    Client-side downscaling is OFF by default (max_pixels=0) on purpose. The
    model's dynamic tiling picks its 448px tile grid partly from image *area*
    (`find_closest_aspect_ratio_optim` ranks candidate grids by area difference),
    so pre-shrinking can land the page on a smaller grid than it deserves and
    quietly cost resolution. Set --max-pixels only when huge scans make the
    request payloads themselves a problem.
    """
    img = to_pil_image(image)
    w, h = img.size
    if max_pixels and w * h > max_pixels:
        scale = math.sqrt(max_pixels / (w * h))
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


# --- Formula cleaning (verbatim from the model repo's eval/hpd_to_markdown.py) ---

_TALL = re.compile(
    r'\\d?frac|\\tfrac|\\cfrac|\\binom|\\sqrt'
    r'|\\sum|\\prod|\\coprod|\\int|\\iint|\\iiint|\\oint'
    r'|\\bigcup|\\bigcap|\\bigoplus|\\bigotimes|\\bigsqcup'
    r'|\\begin\{'
    r'|\\overbrace|\\underbrace|\\overset|\\underset|\\stackrel'
    r'|\\substack|\\atop|\\\\'
)


def _scan_delims(s):
    out = []
    for m in re.finditer(r'\\(left|right)\s*', s):
        dm = re.match(r'\\[a-zA-Z]+|\\.|.', s[m.end():])
        if not dm:
            continue
        out.append({'kind': m.group(1), 'delim': dm.group(0),
                    'start': m.start(), 'end': m.end() + dm.end()})
    return out


def simplify_left_right(s: str) -> str:
    """Downgrade `\\left( ... \\right)` with no tall inner structure to plain `( )`."""
    if '\\left' not in s:
        return s
    stack, pairs = [], []
    for d in _scan_delims(s):
        if d['kind'] == 'left':
            stack.append(d)
        elif stack:
            pairs.append((stack.pop(), d))
    edits = []
    for L, R in pairs:
        if L['delim'] == '(' and R['delim'] == ')' and not _TALL.search(s[L['end']:R['start']]):
            edits.append((L['start'], L['end'], '('))
            edits.append((R['start'], R['end'], ')'))
    for st, en, rep in sorted(edits, key=lambda x: x[0], reverse=True):
        s = s[:st] + rep + s[en:]
    return s


_ELLIPSIS = r'(?:\\dots|\\cdots|\\ldots|\\dotsb|\\dotsc)'
_CLOSER = r'(?:\\right\s*[.\}\]\)]|\\end\s*\{(?:array|matrix|cases|bmatrix|pmatrix|vmatrix|smallmatrix)\})'
_TAIL_WRAP = re.compile(r'^(?P<core>.*?)(?P<wrap>\s*(?:\\\]|\\\)|\$\$))?\s*$', re.DOTALL)


def clean_formula_tail(s: str) -> str:
    """Strip degenerate formula tails (repeated/dangling ellipses, stray `\\quad`)."""
    if not s:
        return s
    m = _TAIL_WRAP.match(s)
    core, wrap = m.group('core'), m.group('wrap') or ''
    prev = None
    while prev != core:
        prev = core
        core = re.sub(r'(' + _ELLIPSIS + r')(?:\s*' + _ELLIPSIS + r')+', r'\1', core)
        core = re.sub(r'(?P<keep>' + _CLOSER + r')\s*(?:\\q?quad\s*)*' + _ELLIPSIS + r'\s*$',
                      lambda mm: mm.group('keep'), core)
        core = re.sub(r'(?:\s*\\q?quad)+\s*' + _ELLIPSIS + r'\s*$', '', core)
        core = re.sub(r'(?:\s*\\q?quad)+\s*$', '', core)
        core = core.rstrip()
    return core + wrap


_OP_MAP = {
    '≈': r'\approx', '≠': r'\neq', '≤': r'\leq', '≥': r'\geq', '×': r'\times',
    '÷': r'\div', '±': r'\pm', '∓': r'\mp', '·': r'\cdot', '∙': r'\cdot',
    '⋅': r'\cdot', '∗': '*', '−': '-', '≡': r'\equiv', '∝': r'\propto',
    '∞': r'\infty', '√': r'\sqrt', '→': r'\to', '≪': r'\ll', '≫': r'\gg',
}
_ARITH_ALLOWED = re.compile(r'^[0-9A-Za-z\s=+\-*/^_().,:;<>|%!\u4e00-\u9fff' + ''.join(_OP_MAP.keys()) + r']+$')
_ARITH_HASOP = re.compile(r'[=+\-*/' + ''.join(_OP_MAP.keys()) + r']')
_KNOWN_FUNCS = {'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'log', 'ln', 'exp',
                'lim', 'max', 'min', 'det', 'mod', 'arcsin', 'arccos', 'arctan', 'sqrt'}
_CJK_RUN = re.compile(r'[\u4e00-\u9fff]+')
_MATH_SPAN = re.compile(r'(\\\[.*?\\\]|\$\$.*?\$\$|\\\(.*?\\\)|\$.*?\$)', re.DOTALL)

WRAP_CJK_IN_ARITH = True


def _convert_unicode_ops(s: str) -> str:
    for k, v in _OP_MAP.items():
        s = s.replace(k, (v + ' ') if v.startswith('\\') else v)
    if WRAP_CJK_IN_ARITH:
        s = _CJK_RUN.sub(lambda m: r'\text{' + m.group(0) + '}', s)
    return re.sub(r'[ \t]{2,}', ' ', s)


def _is_pure_arith_line(line: str) -> bool:
    t = line.strip()
    if not t or '\\(' in t or '\\[' in t or '$' in t or '<' in t:
        return False
    if not WRAP_CJK_IN_ARITH and re.search(r'[\u4e00-\u9fff]', t):
        return False
    if not _ARITH_ALLOWED.match(t) or not _ARITH_HASOP.search(t):
        return False
    return all(w.lower() in _KNOWN_FUNCS for w in re.findall(r'[A-Za-z]{2,}', t))


def normalize_arith(text: str) -> str:
    """Normalize Unicode operators to LaTeX and wrap pure-arithmetic lines as `\\( .. \\)`."""
    if not text:
        return text
    text = _MATH_SPAN.sub(lambda m: _convert_unicode_ops(m.group(0)), text)
    out = []
    for line in text.split('\n'):
        if _is_pure_arith_line(line):
            out.append('\\( ' + _convert_unicode_ops(line.strip()) + ' \\)')
        else:
            out.append(line)
    return '\n'.join(out)


def clean_text(text: str) -> str:
    """Per-block cleaning, matching the official hpd_to_markdown.py steps."""
    text = text.strip()
    text = text.replace('The image is too blurry to recognize any text content.', '').strip()
    text = text.replace(
        "The image contains no text or characters. It is a graphical element (a horizontal "
        "line with a vertical line) and does not contain any chart, graph, or data points "
        "that can be extracted. Therefore, the correct OCR output is an empty string.", ""
    ).strip()
    if not text or text == '[Non-Text]':
        return ''
    if text.startswith('\\[') and not text.endswith('\n\\]'):
        text += '\n\\]'
    if text.startswith('<table>') and not text.endswith('</table>'):
        text += '</table>'
    if '\\[\n' in text and '\\\\' not in text:
        text = text.replace('\\[\n', '\\(').replace('\n\\]', '\\)')
    text = text.replace('\\) \\(', '\\)\n\n\\(')
    if '÷' in text and '\\(' not in text:
        text = '\\( ' + text + ' \\)'
    text = re.sub(r'\\tag\s*\{[^{}]*\}', '', text)
    text = text.replace('\\supset', '\\sqsupset')
    text = simplify_left_right(text)
    text = clean_formula_tail(text)
    return normalize_arith(text)


# --- Block parsing (bbox-preserving; the official script discards bboxes) ---

# type + [bbox], usually followed by <FORK|CHILD|BLOCK>. Container blocks such as
# `list [x1,y1,x2,y2]` may have no trailing tag at all, so the tag is optional.
_BLOCK_HEADER = re.compile(r'([a-zA-Z_]+)\s*\[\s*([-\d.,\s]+)\]\s*(?:<(?:FORK|CHILD|BLOCK)>)?')
_NO_CONTENT_TYPES = {'chart', 'seal'}


def parse_blocks(raw_text: str) -> List[Dict[str, Any]]:
    """Parse the `<BLOCK> <type> [bbox] <CHILD> <content>` stream.

    Returns `[{"type": str, "bbox": [x1,y1,x2,y2] | None, "text": str}, ...]` in the
    model's own reading order. `bbox` is in the model's native 0-1000 NORMALIZED
    coordinate space, not pixels. `text` is empty for container blocks (e.g. `list`,
    the `table` wrapper) and for `chart`/`seal`, matching the official export.
    """
    blocks = []
    for seg in raw_text.split('<BLOCK>')[1:]:
        header_m = _BLOCK_HEADER.match(seg.strip())
        type_m = re.match(r'\s*([a-zA-Z_]+)', seg)
        block_type = type_m.group(1) if type_m else 'unknown'

        bbox = None
        if header_m:
            block_type = header_m.group(1)
            nums = [float(x) for x in re.split(r'[,\s]+', header_m.group(2).strip()) if x]
            if len(nums) == 4:
                bbox = nums

        text = ''
        if block_type.lower() not in _NO_CONTENT_TYPES:
            content_m = re.search(r'<CHILD>(.*)', seg, re.DOTALL)
            if content_m:
                # stop at the next control tag if any leaked through
                raw_content = re.split(r'<(?:FORK|CHILD|BLOCK)>', content_m.group(1))[0]
                text = clean_text(raw_content)

        blocks.append({'type': block_type, 'bbox': bbox, 'text': text})
    return blocks


def blocks_to_markdown(blocks: List[Dict[str, Any]]) -> str:
    """Join the non-empty block texts in reading order into one markdown string."""
    return '\n\n'.join(b['text'] for b in blocks if b['text']).strip()


# --- server lifecycle ---


def server_alive(server: str) -> bool:
    try:
        return requests.get(f"{server}/health", timeout=5).status_code == 200
    except requests.RequestException:
        return False


def wait_for_server(server: str, timeout_s: int, proc: "subprocess.Popen | None" = None):
    logger.info(f"Waiting for server at {server}...")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if server_alive(server):
            logger.info("Server is ready")
            return
        if proc is not None and proc.poll() is not None:
            logger.error(
                f"Spawned vllm serve exited with code {proc.returncode} before becoming ready"
            )
            sys.exit(1)
        time.sleep(10)
    logger.error(f"Server did not become ready within {timeout_s}s")
    sys.exit(1)


def ensure_server(server: str, serve_args_factory, timeout_s: int = 1800):
    """Use a reachable server; otherwise spawn `vllm serve` ourselves; else fail fast.

    Spawning is only attempted for a localhost URL — a remote --server that is
    down is the user's to fix, not ours to shadow with a local model.
    """
    if server_alive(server):
        logger.info(f"Using already-running server at {server}")
        return

    host = urlparse(server).hostname or ""
    if host not in ("127.0.0.1", "localhost", "0.0.0.0"):
        logger.info(f"Remote server {server} not up yet — waiting for it")
        wait_for_server(server, timeout_s)
        return

    if shutil.which("vllm") is None:
        logger.error("No server is running and the `vllm` binary is not on PATH.")
        logger.error(
            "HPD-Parsing needs the vendor's customized vLLM build (dynamic request "
            "forking + P-MTP), which exists in no upstream wheel. Run this script on "
            "the vendor image so it can start the server itself:\n"
        )
        logger.error(RUN_COMMAND)
        logger.error("\n(or start `vllm serve` yourself / pass --server URL of a running endpoint)")
        sys.exit(1)

    serve_args = serve_args_factory()
    logger.info(f"Starting server: {' '.join(serve_args)}")
    proc = subprocess.Popen(serve_args)  # logs interleave with ours on stdout/stderr
    atexit.register(proc.terminate)  # don't leave a GPU server behind on local runs
    wait_for_server(server, timeout_s, proc=proc)


def parse_one(
    server: str,
    image,
    prompt: str,
    max_pixels: int,
    max_tokens: int,
    timeout_s: int,
    retries: int = 2,
) -> str:
    """Parse a single image via the chat completions API. Returns raw model text."""
    b64 = encode_image(image, max_pixels)
    payload = {
        "model": SERVED_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    # Image first, then text — the order the card's example uses.
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                f"{server}/v1/chat/completions", json=payload, timeout=timeout_s
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"request failed after {retries + 1} attempts: {last_err}")


def create_dataset_card(
    source_dataset: str,
    model: str,
    num_samples: int,
    num_errors: int,
    processing_time: str,
    images_per_sec: float,
    concurrency: int,
    max_tokens: int,
    use_fork: bool,
    output_column: str,
    blocks_column: str,
    keep_raw: bool,
    raw_column: str,
    image_column: str = "image",
    split: str = "train",
) -> str:
    """Create a dataset card documenting the parsing run."""
    model_name = model.split("/")[-1]

    # Canonical provenance stamp (see AGENTS.md): Jobs claim gated on JOB_ID, set by HF Jobs in-container.
    on_jobs = os.environ.get("JOB_ID") is not None
    hw = os.environ.get("ACCELERATOR") or ""
    origin = (
        "Produced on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)"
        + (f" (`{hw}`)" if hw else "")
    ) if on_jobs else "Generated"
    jobs_tag = "\n- hf-jobs" if on_jobs else ""
    raw_row = f"\n- `{raw_column}`: the untouched `<BLOCK>/<FORK>/<CHILD>` model output" if keep_raw else ""

    return f"""---
tags:
- ocr
- document-processing
- hpd-parsing
- layout-analysis
- markdown
- uv-script
- generated{jobs_tag}
---

# Document parsing using {model_name} (server mode)

This dataset contains document-parsing results from images in [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using HPD-Parsing, a 1B hierarchical-parallel-decoding parser (94.91 on OmniDocBench v1.6 at 4,752 TPS), served behind an in-job vLLM server with concurrent requests.

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Model**: [{model}](https://huggingface.co/{model})
- **Number of Samples**: {num_samples:,}
- **Failed Requests**: {num_errors:,} (marked `[OCR ERROR]`)
- **Processing Time**: {processing_time}
- **Throughput**: {images_per_sec:.2f} images/sec
- **Processing Date**: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}

### Configuration

- **Mode**: customized vLLM server (`vllm serve`) + concurrent driver, {concurrency} concurrent requests
- **Decoding**: {"hierarchical parallel (fork) + P-MTP speculative decoding" if use_fork else "standard single-trajectory (--no-fork)"}
- **Image Column**: `{image_column}`
- **Dataset Split**: `{split}`
- **Max Output Tokens**: {max_tokens:,}
- **Temperature**: 0.0 (greedy, per model card)

## Dataset Structure

The dataset contains all original columns plus:
- `{output_column}`: the extracted text as markdown, blocks joined in reading order
- `{blocks_column}`: JSON list of the parsed blocks, one entry per region:
  `{{"type": "title|text|table|figure|formula|...", "bbox": [x1, y1, x2, y2], "text": "..."}}`{raw_row}
- `inference_info`: JSON list tracking all OCR models applied to this dataset

### Bounding boxes are normalized, not pixels

`bbox` values are in the model's native **0-1000 normalized** coordinate space (unlike
the pixel-space boxes some other recipes emit). Convert to pixels with:

```python
x_px = x / 1000 * image.width
y_px = y / 1000 * image.height
```

Blocks appear in the model's own reading order; container blocks (e.g. `list`) and
`chart`/`seal` regions carry a bbox with empty `text`.

## Reproduction

{origin} with the [`hpd-parsing-server.py`](https://huggingface.co/datasets/uv-scripts/ocr/raw/main/hpd-parsing-server.py) recipe from [uv-scripts](https://huggingface.co/uv-scripts). HPD-Parsing needs the vendor's customized vLLM build, so the job runs on Paddle's own image (which ships no `uv`, hence the bootstrap):

```bash
hf jobs run --detach --flavor l4x1 -s HF_TOKEN --timeout 2h \\
    {IMAGE_TAG} \\
    -- bash -lc 'pip install -q uv && uv run \\
        https://huggingface.co/datasets/uv-scripts/ocr/raw/main/hpd-parsing-server.py \\
        {source_dataset} <output-dataset>'
```
"""


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    server: Optional[str] = None,
    port: int = DEFAULT_PORT,
    concurrency: int = 32,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    max_pixels: int = 0,
    request_timeout: int = 1800,
    use_fork: bool = True,
    hf_token: str = None,
    split: str = "train",
    max_samples: int = None,
    private: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    output_column: str = "markdown",
    blocks_column: str = "hpd_blocks",
    keep_raw: bool = False,
    raw_column: str = "hpd_raw",
    overwrite: bool = False,
    config: str = None,
    create_pr: bool = False,
    verbose: bool = False,
):
    """Process images from an HF dataset through an HPD-Parsing vLLM server."""

    start_time = datetime.now()

    server = server or f"http://127.0.0.1:{port}"

    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    logger.info(f"Using model: {MODEL} via server {server}")

    logger.info(f"Loading dataset: {input_dataset}")
    dataset = load_dataset(input_dataset, split=split)

    if image_column not in dataset.column_names:
        raise ValueError(
            f"Column '{image_column}' not found. Available: {dataset.column_names}"
        )

    out_cols = [output_column, blocks_column] + ([raw_column] if keep_raw else [])
    dataset = ensure_output_columns_free(dataset, out_cols, overwrite=overwrite)

    if shuffle:
        logger.info(f"Shuffling dataset with seed {seed}")
        dataset = dataset.shuffle(seed=seed)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples")

    # Reuse a reachable server, else spawn `vllm serve` (needs the vendor image's
    # forked vllm binary), else fail fast with the correct command. The serve args are
    # built lazily so that pointing --server at a running endpoint doesn't download the
    # speculative head we would never use.
    ensure_server(
        server,
        lambda: build_serve_args(port, max_model_len, resolve_spec_model()),
    )

    prompt = FORK_PROMPT if use_fork else PLAIN_PROMPT
    logger.info(f"Prompt: {prompt!r}")

    n = len(dataset)
    logger.info(f"Processing {n} images, concurrency {concurrency}")
    markdowns: List[Optional[str]] = [None] * n
    blocks_json: List[Optional[str]] = [None] * n
    raws: List[Optional[str]] = [None] * n
    errors = 0
    done = 0
    inference_start = time.time()
    lock = threading.Lock()

    def worker(i: int) -> None:
        nonlocal errors, done
        try:
            raw = parse_one(
                server,
                dataset[i][image_column],
                prompt,
                max_pixels,
                max_tokens,
                request_timeout,
            )
            blocks = parse_blocks(raw)
            markdowns[i] = blocks_to_markdown(blocks)
            blocks_json[i] = json.dumps(blocks, ensure_ascii=False)
            raws[i] = raw
        except Exception as e:
            logger.error(f"Image {i} failed: {e}")
            markdowns[i] = "[OCR ERROR]"
            blocks_json[i] = json.dumps([])
            raws[i] = ""
            with lock:
                errors += 1
        with lock:
            done += 1
            if done % 25 == 0 or done == n:
                rate = done / max(time.time() - inference_start, 1e-9)
                logger.info(f"{done}/{n} done ({rate:.2f} img/s, {errors} errors)")

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(worker, range(n)))

    inference_secs = time.time() - inference_start
    processing_duration = datetime.now() - start_time
    processing_time_str = f"{processing_duration.total_seconds() / 60:.1f} min"
    images_per_sec = n / inference_secs if inference_secs else 0.0

    logger.info(f"Adding '{output_column}' and '{blocks_column}' columns to dataset")
    dataset = dataset.add_column(output_column, markdowns)
    dataset = dataset.add_column(blocks_column, blocks_json)
    if keep_raw:
        dataset = dataset.add_column(raw_column, raws)

    inference_entry = {
        "model_id": MODEL,
        "model_name": "HPD-Parsing",
        "column_name": output_column,
        "blocks_column": blocks_column,
        # Not pixels — the trap this field exists to prevent.
        "bbox_space": "normalized_0_1000",
        "timestamp": datetime.now().isoformat(),
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "max_pixels": max_pixels,
        "fork_decoding": use_fork,
        "prompt": prompt,
        "mode": "vllm-server",
        "concurrency": concurrency,
    }

    if "inference_info" in dataset.column_names:
        logger.info("Updating existing inference_info column")

        def update_inference_info(example):
            try:
                existing_info = (
                    json.loads(example["inference_info"])
                    if example["inference_info"]
                    else []
                )
            except (json.JSONDecodeError, TypeError):
                existing_info = []
            existing_info.append(inference_entry)
            return {"inference_info": json.dumps(existing_info)}

        dataset = dataset.map(update_inference_info)
    else:
        logger.info("Creating new inference_info column")
        inference_list = [json.dumps([inference_entry])] * len(dataset)
        dataset = dataset.add_column("inference_info", inference_list)

    # Push to hub with retry and XET fallback
    logger.info(f"Pushing to {output_dataset}")
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                logger.warning("Disabling XET (fallback to HTTP upload)")
                os.environ["HF_HUB_DISABLE_XET"] = "1"
            dataset.push_to_hub(
                output_dataset,
                private=private,
                token=HF_TOKEN,
                max_shard_size="500MB",
                **({"config_name": config} if config else {}),
                create_pr=create_pr,
                commit_message=f"Add {MODEL} parsing results ({len(dataset)} samples, server mode)"
                + (f" [{config}]" if config else ""),
            )
            break
        except Exception as e:
            logger.error(f"Upload attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                delay = 30 * (2 ** (attempt - 1))
                logger.info(f"Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error("All upload attempts failed. Parsing results are lost.")
                sys.exit(1)

    logger.info("Creating dataset card")
    card_content = create_dataset_card(
        source_dataset=input_dataset,
        model=MODEL,
        num_samples=len(dataset),
        num_errors=errors,
        processing_time=processing_time_str,
        images_per_sec=images_per_sec,
        concurrency=concurrency,
        max_tokens=max_tokens,
        use_fork=use_fork,
        output_column=output_column,
        blocks_column=blocks_column,
        keep_raw=keep_raw,
        raw_column=raw_column,
        image_column=image_column,
        split=split,
    )

    card = DatasetCard(card_content)
    card.push_to_hub(output_dataset, token=HF_TOKEN)

    logger.info("Done! HPD-Parsing server-mode processing complete.")
    logger.info(f"Dataset available at: https://huggingface.co/datasets/{output_dataset}")
    logger.info(f"Processing time: {processing_time_str}")
    logger.info(
        f"Throughput: {images_per_sec:.2f} images/sec "
        f"(inference only, excl. dataset load/push; {errors} errors)"
    )

    if verbose:
        import importlib.metadata

        logger.info("--- Resolved package versions ---")
        for pkg in ["datasets", "huggingface-hub", "pyarrow", "pillow", "requests"]:
            try:
                logger.info(f"  {pkg}=={importlib.metadata.version(pkg)}")
            except importlib.metadata.PackageNotFoundError:
                logger.info(f"  {pkg}: not installed")
        try:
            out = subprocess.run(
                ["vllm", "--version"], capture_output=True, text=True, timeout=60
            )
            logger.info(f"  vllm (server) == {out.stdout.strip() or out.stderr.strip()}")
        except Exception as e:
            logger.info(f"  vllm (server): not queryable ({e})")
        logger.info("--- End versions ---")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 70)
        print("HPD-Parsing Document Parsing (customized vLLM server mode)")
        print("=" * 70)
        print("\n1B hierarchical-parallel-decoding parser — 94.91 OmniDocBench v1.6")
        print("at 4,752 TPS (2.62x the fastest prior parser).")
        print("\nOutputs markdown in reading order PLUS a structured hpd_blocks")
        print("column: type + bbox + text per region (bbox = 0-1000 normalized).")
        print("\nNeeds the vendor image (forked vLLM with request forking):")
        print(f"  {IMAGE_TAG}")
        print("\nExamples:")
        print("\n1. Basic run (spawns the server itself):")
        print("   uv run hpd-parsing-server.py input-dataset output-dataset")
        print("\n2. Test with a small sample:")
        print("   uv run hpd-parsing-server.py large-dataset test --max-samples 10 --shuffle")
        print("\n3. Standard (non-fork) decoding, for an A/B:")
        print("   uv run hpd-parsing-server.py docs results --no-fork")
        print("\n4. On HF Jobs:")
        print(f"   {RUN_COMMAND}")
        print("\nFor full help: uv run hpd-parsing-server.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Document parsing using HPD-Parsing via an in-job customized vLLM server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run hpd-parsing-server.py my-docs analyzed-docs
  uv run hpd-parsing-server.py large-dataset test --max-samples 50 --shuffle
  uv run hpd-parsing-server.py docs results --keep-raw
See the module docstring for the full `hf jobs uv run` command (needs --image).
        """,
    )

    parser.add_argument("input_dataset", help="Input dataset ID from Hugging Face Hub")
    parser.add_argument("output_dataset", help="Output dataset ID for Hugging Face Hub")
    parser.add_argument(
        "--image-column",
        default="image",
        help="Column containing images (default: image)",
    )
    parser.add_argument(
        "--server",
        help=f"vLLM server base URL (default: in-job localhost on --port, i.e. {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port for the spawned server (default: {DEFAULT_PORT}, the vendor default)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Concurrent parsing requests (default: 32; vLLM queues excess internally, "
        "so this mainly needs to be high enough to keep continuous batching fed)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum tokens to generate (default: {DEFAULT_MAX_TOKENS}, the model card value)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
        help=f"Server context length (default: {DEFAULT_MAX_MODEL_LEN}, the card/entrypoint "
        "value; the LM itself allows 40960)",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=0,
        help="Downscale images above this pixel count client-side before upload "
        "(default: 0 = off; the model's tile-grid choice is area-sensitive, so "
        "pre-shrinking can cost resolution — use only to shrink huge payloads)",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=1800,
        help="Per-request timeout in seconds (default: 1800)",
    )
    parser.add_argument(
        "--no-fork",
        action="store_true",
        help="Use the standard single-trajectory prompt ('document parsing.') instead of "
        "hierarchical parallel decoding ('document parsing with fork.'). Slower; for A/B only",
    )
    parser.add_argument("--hf-token", help="Hugging Face API token")
    parser.add_argument(
        "--split", default="train", help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to process (for testing)",
    )
    parser.add_argument(
        "--private", action="store_true", help="Make output dataset private"
    )
    parser.add_argument(
        "--config",
        help="Config/subset name when pushing to Hub (for benchmarking multiple models in one repo)",
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Create a pull request instead of pushing directly (for parallel benchmarking)",
    )
    parser.add_argument(
        "--shuffle", action="store_true", help="Shuffle dataset before processing"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling (default: 42)"
    )
    parser.add_argument(
        "--output-column",
        default="markdown",
        help="Column name for the markdown output (default: markdown)",
    )
    parser.add_argument(
        "--blocks-column",
        default="hpd_blocks",
        help="Column name for the structured block JSON (default: hpd_blocks)",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Also write the untouched <BLOCK>/<FORK>/<CHILD> stream to --raw-column",
    )
    parser.add_argument(
        "--raw-column",
        default="hpd_raw",
        help="Column name for the raw model output when --keep-raw (default: hpd_raw)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output columns if they already exist in the input dataset "
        "(default: error out to avoid clobbering existing columns).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log resolved package versions after processing (useful for pinning deps)",
    )

    args = parser.parse_args()

    if args.max_tokens > args.max_model_len:
        parser.error(
            f"--max-tokens ({args.max_tokens}) must be <= --max-model-len ({args.max_model_len})"
        )

    main(
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        image_column=args.image_column,
        server=args.server,
        port=args.port,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        max_pixels=args.max_pixels,
        request_timeout=args.request_timeout,
        use_fork=not args.no_fork,
        hf_token=args.hf_token,
        split=args.split,
        max_samples=args.max_samples,
        private=args.private,
        shuffle=args.shuffle,
        seed=args.seed,
        output_column=args.output_column,
        blocks_column=args.blocks_column,
        keep_raw=args.keep_raw,
        raw_column=args.raw_column,
        overwrite=args.overwrite,
        config=args.config,
        create_pr=args.create_pr,
        verbose=args.verbose,
    )
