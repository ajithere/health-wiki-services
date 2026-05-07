# Service 2 — Synthesizer

Watches `raw_extracted/incoming/` for new files produced by Service 1.
Calls an LLM to generate wiki page operations, then writes or stages them.
Maintains `index.md` automatically on every ingest.

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy the env template and fill in your values
cp .env.template .env
# Edit .env: set BASE_DIR, LLM_PROVIDER, LLM_MODEL, and your API key

# 3. Create folder structure and starter index.md
python src/synthesizer.py setup
python src/synthesizer.py init-index
```

---

## LLM providers

Set `LLM_PROVIDER` in `.env` to one of:

| Provider | Model example | Requires |
|---|---|---|
| `anthropic` | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| `openai` | `gpt-4o` | `OPENAI_API_KEY` |
| `google` | `gemini-1.5-pro` | `GOOGLE_API_KEY` |
| `ollama` | `llama3` | Ollama running locally |
| `openai-compatible` | `llama3-70b` | `OPENAI_API_KEY` + `OPENAI_BASE_URL` |

Switch providers by changing two lines in `.env` — no code changes.

**Ollama (fully local, free, private):**
```bash
# Install Ollama from https://ollama.com
ollama pull llama3
# In .env:
# LLM_PROVIDER=ollama
# LLM_MODEL=llama3
```

---

## Usage

### Watch mode (recommended)
```bash
python src/synthesizer.py
```
Automatically processes any `.md` file dropped into `raw_extracted/incoming/`.

### Single file
```bash
python src/synthesizer.py file path/to/extracted.md
```

### Batch (process all files in a folder)
```bash
python src/synthesizer.py batch
python src/synthesizer.py batch path/to/raw_extracted/incoming/
```

---

## Staging

When `STAGING_ENABLED=true` in `.env`, wiki writes go to `staging/`
first instead of `wiki/`. Review them before committing to the wiki.

```bash
# See what's staged
python src/synthesizer.py show-staged

# Approve all staged files (moves them to wiki/)
python src/synthesizer.py approve

# Approve without prompt (for automation)
python src/synthesizer.py approve --yes

# Reject all staged files
python src/synthesizer.py reject
```

Set `STAGING_AUTO_APPROVE=true` to bypass the review gate entirely.

---

## Index

`wiki/index.md` is automatically created and maintained on every ingest.
It tracks:

- Reports ingested per person with last report date
- All biomarker pages as wikilinks
- Current status snapshot per person (latest key values)
- Conditions, medications, and imaging studies

**To create a fresh index.md** (first run, or after deleting it):
```bash
python src/synthesizer.py init-index
```

The index is updated as part of every ingest — you never need to
edit it manually. Every LLM operation set includes an `update_index`
operation that surgically updates only the relevant sections.

---

## Lint

Run a health check on the wiki at any time. Read-only — no files
are modified.

```bash
# Print report to console
python src/synthesizer.py lint

# Print and save report to wiki/insights/lint-YYYY-MM-DD.md
python src/synthesizer.py lint --save
```

**What lint checks:**

| Check | What it flags |
|---|---|
| `CONFLICT` | Biomarker rows marked CONFLICT — same date from two sources |
| `INDEX` | index.md missing entirely |
| `INDEX-SYNC` | Pages that exist but are not listed in index.md |
| `MISSING-PAGE` | Tracked biomarkers with no wiki page yet |
| `GAP` | Readings more than 12 months apart in a biomarker page |
| `SINGLE-READING` | Biomarker pages with only one data point (no trend) |
| `ORPHAN` | Pages with no inbound wikilinks |

Run lint periodically — after a batch ingest, or whenever you want
a health check before sharing data with a doctor.

---

## Biomarkers

`config/biomarkers.yaml` controls which parameters are tracked and how
raw lab names are mapped to canonical wiki filenames.

**Add a new tracked parameter:**
```yaml
track:
  - my-new-biomarker
```

**Add an alias for a messy lab name:**
```yaml
aliases:
  ldl-cholesterol:
    - LDL
    - LDL-C
    - Your Lab's Custom Name Here
```

---

## Low confidence files

If Service 1 set `confidence: low` in a file's frontmatter (poor OCR,
sparse extraction), the synthesizer skips the LLM call and copies the
file to `raw_extracted/needs-review/` for manual inspection.

---

## What the synthesizer produces per file type

| content_type | Pages created/updated |
|---|---|
| `lab-report` | `wiki/reports/` + `wiki/biomarkers/` (one row per tracked param) |
| `imaging-report-echo` | `wiki/imaging/` + `wiki/biomarkers/ef-percentage` |
| `imaging-report-cac` | `wiki/imaging/` + `wiki/biomarkers/cac-score` |
| `imaging-report-usg` | `wiki/imaging/` |
| `imaging-report-radiology` | `wiki/imaging/` |
| `smart-report` | `wiki/reports/` + `wiki/trends/` |
| `trend-data` | `wiki/biomarkers/` (multiple parameters) |
| `clinical-note` | `wiki/reports/` |
| All types | `wiki/people/<person>.md` + `wiki/log.md` + `wiki/index.md` |

---

## Full command reference

```bash
python src/synthesizer.py                    # watch mode (default)
python src/synthesizer.py setup              # create folder structure
python src/synthesizer.py init-index         # create starter index.md
python src/synthesizer.py file <path>        # process single file
python src/synthesizer.py batch [folder]     # process all files in folder
python src/synthesizer.py show-staged        # review staged writes
python src/synthesizer.py approve [--yes]    # commit staged to wiki
python src/synthesizer.py reject             # discard staged writes
python src/synthesizer.py lint [--save]      # health check wiki
```

---

## What this service does NOT do

- Extract content from raw files (that is Service 1's job)
- Directly diagnose or recommend treatment
- Run correlation sweeps automatically (on-demand only —
  trigger phrase: "Run correlation sweep")
- Modify files in `raw_input/` or `raw_extracted/<year>/`
