# Serve Unlimited-OCR as a live endpoint on HF Jobs

The OCR recipes in this folder run as batch jobs (dataset in → dataset out). To call a model
interactively, from an agent, or with ad-hoc concurrent requests, you can instead run it as a
temporary HTTP endpoint. [HF Jobs serving](https://huggingface.co/docs/hub/jobs-serving) exposes a
port on a GPU Job, giving an OpenAI-compatible endpoint that runs until the job is cancelled or its
`--timeout` is reached.

This is a worked example for [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR)
(3B, MIT, based on DeepSeek-OCR; supports multi-page parsing in a single request). Two server
options below: **vLLM** on Baidu's official image (the newer official path, OpenAI-compatible), or
**SGLang** on the stock image with the model's own wheel. Either gives an OpenAI-compatible endpoint.

> **Single-image vs multi-page — pick the engine by task:**
> - **Single-page** OCR (one image → markdown): both engines work. For a whole corpus, the batch
>   recipe [`unlimited-ocr-vllm.py`](unlimited-ocr-vllm.py) (offline vLLM, resumable, no network) is
>   the better fit than a client loop; for interactive/agent use, serve with **vLLM (Option A)**.
> - **Multi-page / long-horizon** parsing (the model's headline feature): use **SGLang (Option B)** —
>   **validated working 2026-06-28** (a clean 2-page doc read back both pages verbatim, `<PAGE>`-separated;
>   it needs the pinned image + a100 setup in Option B). This is the model authors' documented multi-page
>   path (`images_config`). The **vLLM** integration ([PR #46564](https://github.com/vllm-project/vllm/pull/46564))
>   is single-image only — its benchmark is single-page OmniDocBench, the authors' multi-page smoke test
>   only confirmed the R-SWA *mechanism* (not OCR quality), and in our tests vLLM multi-image came back
>   garbled (hallucinated text). So don't use Option A or the batch recipe for multi-page — use Option B.

## 1. Start the server

### Option A — vLLM (official image) · single-image only

vLLM support landed upstream; Baidu ships a dedicated image (the architecture isn't in a stable pip
wheel yet). Use the default `:unlimited-ocr` tag on L4/A100, or `:unlimited-ocr-cu129` on Hopper.
**Validated for single-image OCR only** — runs on `l4x1`, no fa3/Hopper requirement. (For
**multi-page**, use Option B; vLLM multi-image OCR isn't demonstrated upstream and came back garbled
in our tests.)

```bash
hf jobs run --detach --expose 8000 --flavor l4x1 -s HF_TOKEN --timeout 30m \
  vllm/vllm-openai:unlimited-ocr -- \
  vllm serve baidu/Unlimited-OCR --served-model-name Unlimited-OCR \
    --trust-remote-code --max-model-len 32768 --host 0.0.0.0 --port 8000 \
    --logits_processors vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor \
    --no-enable-prefix-caching --mm-processor-cache-gb 0
```

Per-request, vLLM takes the no-repeat n-gram knobs via `vllm_xargs` and needs `skip_special_tokens`
off (it has no `images_config` — that's an SGLang param):

```python
r = client.chat.completions.create(
    model="Unlimited-OCR",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "<image>document parsing."},  # literal <image> prefix is required
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
    ]}],
    temperature=0,
    extra_body={"skip_special_tokens": False, "vllm_xargs": {"ngram_size": 35, "window_size": 128}},
)
```

### Option B — SGLang (model's own build) · supports multi-page

The model also ships its own SGLang build, installed at startup from a 12 MB wheel. **This is the
working path for multi-page / long-horizon parsing** (§3) — the model authors' documented route
(`images_config`). Two pins matter (both learned the hard way, 2026-06-28):
- **Pin the image to `lmsysorg/sglang:v0.5.10.post1`** — *not* `:latest`. `:latest` drifted to torch
  2.11 / cu130, incompatible with the wheel (torch 2.9.1 / cuda-python 12.9); v0.5.10.post1 is the last
  release that matches the wheel exactly.
- **Run on `a100-large` with `--attention-backend flashinfer`, not `h200`/`fa3`.** `fa3` needs a Hopper
  GPU, but HF's `h200` nodes currently fail GPU init with `CUDA error 802: system not yet initialized`
  (3/3 attempts) — an infra issue, not the model. `a100` + `flashinfer` sidesteps it and works.

```bash
hf jobs run --detach --expose 10000 --flavor a100-large -s HF_TOKEN --timeout 30m \
  lmsysorg/sglang:v0.5.10.post1 -- \
  bash -lc 'pip install --no-deps https://github.com/baidu/Unlimited-OCR/raw/main/wheel/sglang-0.0.0.dev11416+g92e8bb79e-py3-none-any.whl \
    && pip install -q kernels==0.11.7 \
    && python -m sglang.launch_server --model baidu/Unlimited-OCR --served-model-name Unlimited-OCR \
       --attention-backend flashinfer --page-size 1 --mem-fraction-static 0.85 --context-length 32768 \
       --enable-custom-logit-processor --disable-overlap-schedule --skip-server-warmup \
       --host 0.0.0.0 --port 10000'
```

Notes:
- `--` before `bash` is required, or the CLI parses `-lc` as its own flags.
- `--timeout` stops the endpoint (and billing) at the deadline; `hf jobs cancel <id>` stops it earlier.
- **Validated 2026-06-28** on `a100-large`: server came up, single-image and multi-page both read
  correctly (a clean 2-page doc returned both pages verbatim, `<PAGE>`-separated). The model card's
  "official" backend is `fa3` on Hopper for exact R-SWA — switch back to `--attention-backend fa3
  --flavor h200` once the h200 `802` infra issue clears; `flashinfer` on `a100` is the working fallback.
- Follow startup with `hf jobs logs -f <id>`; ready at `The server is fired up` / `Application startup
  complete` (a few minutes cold; the wheel + model download dominate).

The client examples below use the **SGLang** request format (`images_config` in `extra_body`,
port 10000). The single-image call (§2) also works on the vLLM server — just use the Option A
`extra_body` and your exposed port. **Multi-page (§3) is SGLang-only.**

## 2. Call it (OpenAI client; HF token as the API key)

The exposed port is at `https://<job_id>--10000.hf.jobs`; the OpenAI base URL is that plus `/v1`.

```python
import base64, os
from openai import OpenAI

client = OpenAI(base_url="https://<job_id>--10000.hf.jobs/v1", api_key=os.environ["HF_TOKEN"])
img = base64.b64encode(open("page.jpg", "rb").read()).decode()

r = client.chat.completions.create(
    model="Unlimited-OCR",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "document parsing."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
    ]}],
    temperature=0,
    extra_body={"images_config": {"image_mode": "gundam"}},  # "gundam" (crop-tiling) or "base"
)
print(r.choices[0].message.content)
```

Output is layout-grounded markdown: each block is tagged `<|det|>type [x1,y1,x2,y2]<|/det|> text`,
with coordinates normalized to 0–1000. Remove the tags for plain text
(`re.sub(r'<\|det\|>.*?<\|/det\|>', '', text)`) or keep them for structure.

## 3. Multi-page / PDF — SGLang (Option B) only

> ✅ This requires the **SGLang** server (Option B) — **validated working 2026-06-28** (a clean 2-page
> doc read back both pages verbatim, `<PAGE>`-separated). The `images_config`/`image_mode` param it
> relies on is an SGLang feature; the **vLLM** integration ignores it and returned garbled (hallucinated)
> multi-image output in our tests, and vLLM multi-page OCR isn't demonstrated upstream either
> ([PR #46564](https://github.com/vllm-project/vllm/pull/46564) benchmarks single-page only). This flow
> follows the model card's multi-page example.

Send multiple page images in one request with the `Multi page parsing.` prompt and `image_mode="base"`:

```python
parts = [{"type": "text", "text": "Multi page parsing."}]
for page_png in page_images:            # e.g. PDF pages rendered with pymupdf at ~150 dpi
    b64 = base64.b64encode(open(page_png, "rb").read()).decode()
    parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

r = client.chat.completions.create(
    model="Unlimited-OCR",
    messages=[{"role": "user", "content": parts}],
    temperature=0, max_tokens=16384,
    extra_body={"images_config": {"image_mode": "base"}},
)
```

Pages are separated by `<PAGE>`; tables are returned as HTML and equations as LaTeX, with reading
order preserved across pages. The context length is 32k tokens, so split longer documents.

## 4. Concurrency

SGLang batches concurrent requests, so a client can send many requests in parallel to one endpoint;
the upstream [`infer.py`](https://github.com/baidu/Unlimited-OCR/blob/main/infer.py) uses a
`ThreadPoolExecutor` at `concurrency=8`. For a large corpus, a batch job that runs next to the data
(resumable, no network transfer) is usually a better fit than a client-to-endpoint loop.

## 5. Stop it

```bash
hf jobs cancel <job_id>
```

Billing is per-minute for the GPU flavor plus a small flat fee for the exposed port; scheduling time
is not billed. Run `hf jobs hardware` for current flavors and prices.
