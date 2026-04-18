# Quick Start

## Installation

```bash
git clone https://github.com/metazen11/mlxz && cd mlxz
uv sync --all-extras
```

## Verify Environment

```bash
mlxz doctor
```

## Start Serving

```bash
mlxz serve mlx-community/Llama-3.1-8B-Instruct-4bit
```

## Query

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama","messages":[{"role":"user","content":"Hello!"}]}'
```

## With OpenAI SDK

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="llama",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Benchmarking

```bash
mlxz bench --url http://127.0.0.1:8000
```
