# Serve Unlimited-OCR as a live endpoint on HF Jobs

The OCR recipes in this folder run as batch jobs (dataset in → dataset out). To call a model
interactively, from an agent, or with ad-hoc concurrent requests, you can instead run it as a
temporary HTTP endpoint. [HF Jobs serving](https://huggingface.co/docs/hub/jobs-serving) exposes a
port on a GPU Job, giving an OpenAI-compatible endpoint that runs until the job is cancelled or its
`--timeout` is reached.

This is a worked example for [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR)
(3B, MIT, based on DeepSeek-OCR; supports multi-page parsing in a single request). The model ships
its own SGLang build, so it runs on the stock `lmsysorg/sglang` image with the 12 MB wheel
installed at startup; no custom image is required.

## 1. Start the server

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
- `--` before `bash` is required, or the CLI parses `-lc` as its own flags.
- `--timeout` stops the endpoint (and billing) at the deadline; `hf jobs cancel <id>` stops it earlier.
- `fa3` requires a Hopper GPU (e.g. `h200`). The model is small, so the attention backend, not GPU
  memory, determines the flavor. Run `hf jobs hardware` for available flavors.
- Follow startup with `hf jobs logs -f <id>`; the server is ready at `Application startup complete`
  (about 3 minutes from a cold start).

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

## 3. Multi-page / PDF

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
