# n8n Scout Templates

These are workflow files for n8n that scrape trending short-form video per
client niche and drop the best clips into `references/inbox/<client>/`.

**You do not need to install n8n right now.** These files sit here waiting for
when you decide to set it up. Until then, you ingest reference clips manually:

```bash
# Drop an mp4 + sidecar.json into references/inbox/<client>/, then:
python -m clip_pipeline.rag.ingest <client>
```

## What's in this folder

| File | Purpose |
|---|---|
| `scout_template.json` | One n8n workflow that scrapes trending content for ONE niche and drops top picks into the inbox. Copy + edit per niche. |
| `README.md` | This file. |

## When you're ready to use it

### 1. Install n8n locally

Pick one:

```bash
# Docker (if you already use Docker Desktop)
docker run -it --rm \
  --name n8n \
  -p 5678:5678 \
  -v ~/.n8n:/home/node/.n8n \
  -v /Users/donaldk/Desktop/clip-pipeline:/clip-pipeline \
  docker.n8n.io/n8nio/n8n

# OR npx (no Docker)
npx n8n
```

Open `http://localhost:5678`, set a username/password once.

### 2. Import the template

In n8n's UI: **Workflows → Import from File** → select `scout_template.json`.

### 3. Customize per niche

Edit these node values in the imported workflow:

| Node | What to change |
|---|---|
| **Scrape Trending** | Apify token + the `hashtags` list for your niche. Or replace this node with an `Execute Command` running yt-dlp. |
| **Engagement Filter** | `views >= 50000`, `likes/views >= 0.06`, `duration <= 30`. Tune per niche. |
| **Write MP4 / Sidecar** | The absolute path. Change `/Users/donaldk/Desktop/clip-pipeline/references/inbox/fitness_creator_v1/` to your project path + your client name. |
| **Build Sidecar** | `niche` and `client` values. |
| **Trigger Python Ingest** | The absolute python path + the client name. Or remove this node if you'd rather ingest manually. |

### 4. Activate

In n8n: toggle the workflow to **Active**. It runs daily at 6am.

### 5. One workflow per niche

For each client niche, duplicate the workflow and change:
- The hashtags / search terms in **Scrape Trending**
- The folder name in **Write MP4** and **Write Sidecar**
- The client name in **Build Sidecar** and **Trigger Python Ingest**

Each niche is independent. You can run 5 niches in parallel without changes to the Python repo.

## Sidecar JSON schema (what Python expects)

Each `<post_id>.json` next to its `<post_id>.mp4` should look like:

```json
{
  "niche": "fitness_creator",
  "source_url": "https://www.tiktok.com/@example/video/...",
  "client": "fitness_creator_v1",
  "views": 412000,
  "likes": 38000,
  "saves": 6200,
  "shares": 2100,
  "posted_at": "2026-05-13T18:24:00Z"
}
```

The Python `rag/ingest.py` reads this for engagement stats and the niche tag.

## Free-tier scraper alternative

If you don't want to pay for Apify, replace the **Scrape Trending** node with
an `Execute Command` node calling yt-dlp:

```bash
yt-dlp --dump-json --no-download --playlist-end 50 \
  "https://www.tiktok.com/tag/fitness"
```

Then parse the JSON output and continue the workflow. Free, no API key needed.
yt-dlp's output is line-delimited JSON; n8n's `Code` node can parse it.

## Manual workflow (no n8n at all)

If you skip n8n entirely, this works:

```bash
# 1. Copy any short-form MP4 you consider a good reference:
mkdir -p references/inbox/fitness_creator_v1
cp /path/to/great_clip.mp4 references/inbox/fitness_creator_v1/great_clip_001.mp4

# 2. Write the sidecar by hand:
cat > references/inbox/fitness_creator_v1/great_clip_001.json <<EOF
{
  "niche": "fitness_creator",
  "client": "fitness_creator_v1",
  "source_url": "https://www.tiktok.com/...",
  "views": 412000,
  "likes": 38000,
  "saves": 6200,
  "shares": 2100,
  "posted_at": "2026-05-15T00:00:00Z"
}
EOF

# 3. Ingest:
python -m clip_pipeline.rag.ingest fitness_creator_v1
```

The Python side doesn't care whether n8n or you put the files there. Same result.
