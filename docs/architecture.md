# Architecture

See `CLAUDE.md` for the full architecture overview, data models, and build order.

## Inference Backend Switching

Both Ollama (dev) and vLLM (prod) expose an OpenAI-compatible API at `/v1`.
`app/config.py` reads `INFERENCE_BACKEND` from the active `.env` file and
returns a pre-configured `OpenAI` client via `get_inference_client()`.

Switching environments:
```bash
ln -sf .env.mac .env   # MacBook Pro dev
ln -sf .env.dgx .env   # DGX Sparc prod
```

No code changes required — only the `.env` symlink changes.
