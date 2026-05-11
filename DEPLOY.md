# Deploy to Fly.io

## Prerequisites

- `flyctl` installed: `curl -L https://fly.io/install.sh | sh`
- A Fly.io account (free, requires credit card for verification)
- Supabase Postgres with `pgvector` enabled, `init_db.sql` already applied,
  and `chunks` populated via `python -m scripts.index`
- Upstash Redis (TLS endpoint, `rediss://...`)
- OpenRouter API key with at least $1 balance (or use `:free` models)
- Langfuse Cloud project (optional but expected by the assignment)

## Steps

### 1. Log in and create the app

```bash
fly auth login
fly launch --no-deploy --copy-config --name rag-api-homework
```

`--copy-config` keeps our existing `fly.toml`; just confirm prompts for org
and region (`fra` matches the toml).

### 2. Set secrets

These are read by `app/config.py` at runtime via Pydantic `Settings`.

```bash
fly secrets set \
  OPENROUTER_API_KEY='sk-or-v1-...' \
  DATABASE_URL='postgresql://postgres.<ref>:<password>@aws-1-eu-central-1.pooler.supabase.com:6543/postgres?sslmode=require' \
  REDIS_URL='rediss://default:<token>@<host>.upstash.io:6379' \
  LANGFUSE_PUBLIC_KEY='pk-lf-...' \
  LANGFUSE_SECRET_KEY='sk-lf-...' \
  LANGFUSE_HOST='https://cloud.langfuse.com'
```

### 3. Deploy

```bash
fly deploy
```

The first build downloads sentence-transformers weights (~80MB) and bakes them
into the image so cold starts are fast. Expect ~5–7 minutes for the first
deploy, ~2 minutes for subsequent ones.

### 4. Smoke-test the public URL

```bash
APP=https://rag-api-homework.fly.dev

curl $APP/health

curl -N -X POST $APP/chat/stream \
  -H "X-API-Key: demo-free" \
  -H "Content-Type: application/json" \
  -d '{"message":"Why store config in environment variables?"}'

curl -H "X-API-Key: demo-free" $APP/usage/today
curl -H "X-API-Key: demo-free" "$APP/usage/breakdown?hours=1"
```

### 5. Maintenance

```bash
fly logs              # tail logs
fly status            # machine state
fly ssh console       # exec into the running container
fly secrets list      # list secret names (values are not shown)
fly scale memory 2048 # bump memory if torch OOMs on first inference
```

### Re-indexing the document

`POST /index/rebuild` is scaffolded in the README but not implemented in this
homework — to re-index after editing `data/source.md`, run the indexer
locally with the production `DATABASE_URL`:

```bash
DATABASE_URL='<supabase pooler url>' python -m scripts.index
```

This is one-shot and idempotent (TRUNCATE + INSERT).

## Common issues

- **"Tenant or user not found"** — Supabase pooler hostname is wrong. The
  prefix is `aws-0-` OR `aws-1-` depending on which AZ you got assigned;
  copy the URL straight from the Supabase Connect modal.
- **Direct DB host doesn't resolve** — Supabase free tier exposes the
  direct host (`db.<ref>.supabase.co`) over IPv6 only. Use the pooler URL
  on port 6543 (transaction) or 5432 (session).
- **Streaming hangs after a few seconds** — make sure `auto_stop_machines`
  is `false` in `fly.toml`, otherwise Fly closes the connection during
  long SSE streams.
- **OOM at startup** — sentence-transformers loads torch which needs
  ~600MB. Bump VM memory: `fly scale memory 2048`.
