# Serve Unlimited-OCR as a live endpoint on HF Jobs

Most recipes here run OCR as a **batch** job (dataset in → dataset out). Sometimes you'd rather
have a **live endpoint** instead — to poke at a model interactively, point an agent at it, or fan
out a quick concurrent batch. [HF Jobs serving](https://huggingface.co/docs/hub/jobs-serving) lets
you do that: expose a port on a GPU Job and you get a temporary, OpenAI-compatible endpoint that
bills by the minute and disappears when the job stops.

This is a worked example for **[baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR)**
(3B, MIT, built on DeepSeek-OCR; one-shot long-horizon multi-page parsing). It ships its own SGLang
build, so we serve it on the stock `lmsysorg/sglang` image and overlay the 12 MB wheel at startup —
no custom image to build.

## 1. Start the server (one command)

```bash
hf jobs run --detach --expose 10000 --flavor h200 -s HF_TOKEN --timeout 30m \
  lmsysorg/sglang:latest -- \
  bash -lc 'pip install --no-deps https://github.com/baidu/Unlimited-OCR/raw/main/wheel/sglang-0.0.0.dev11416+g92e8bb79e-py3-none-any.whl \
    && pip install -q kernels==0.11.7 \
    && python -m sglang.launch_server --model baidu/Unlimited-OCR --served-model-name Unlimited-OCR \
       --attention-backend fa3 --page-size 1 --mem-fraction-static 0.8 --context-length 32768 \
       --enable-custom-logit-processor --disable-overlap-schedule --skip-server-warmup \
       --host 0.0.0.0 --port 10000'
```

Notes:
- **`--`** before `bash` is required — otherwise the CLI parses `-lc` as its own flags.
- **`--timeout 30m`** bounds cost: the endpoint (and billing) auto-stops at the deadline.
  `hf jobs cancel <id>` stops it sooner.
- **`--flavor h200`** because the model's `fa3` attention backend needs a Hopper GPU. The model is
  small, so fa3 (not GPU memory) is what dictates the flavor. (`hf jobs hardware` lists the options.)
- Watch it come up with `hf jobs logs -f <id>`; ready at `Application startup complete` (~3 min).

## 2. Call it (any OpenAI client; your HF token is the API key)

The exposed port is at `https://<job_id>--10000.hf.jobs` (base URL = `…/v1`).

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

Output is **layout-grounded** markdown — each block tagged `<|det|>type [x1,y1,x2,y2]<|/det|> text`
with coords normalized 0–1000. Strip the tags for plain text
(`re.sub(r'<\|det\|>.*?<\|/det\|>', '', text)`); keep them for structure.

## 3. Multi-page / PDF (the "Unlimited" path)

Send several page images in **one** request with `Multi page parsing.` + `image_mode="base"`:

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

Pages are `<PAGE>`-separated, with tables as HTML, equations as LaTeX, and reading order preserved
across pages. Context is 32k tokens, so chunk very long documents.

## 4. Batch via concurrency (point an agent at it)

SGLang batches concurrent requests, so an agent/script can fire many async requests at the running
endpoint (the upstream [`infer.py`](https://github.com/baidu/Unlimited-OCR/blob/main/infer.py) runs
a `ThreadPoolExecutor` at `concurrency=8`). For a *very* large corpus, a co-located batch job
(compute next to the data, resumable, no network egress) is more robust — but endpoint + async is
great for interactive and agent-driven runs.

## 5. Stop it

```bash
hf jobs cancel <job_id>
```

> Billing is per-minute on top of the GPU flavor, plus a small flat fee for the exposed port;
> scheduling time is free. Run `hf jobs hardware` for current flavors and prices. A short session
> is roughly the cost of a coffee.
