# Service 1 — Extractor

Watches `raw_input/incoming/` for new files, extracts content to
structured markdown in `raw_extracted/incoming/`, then archives
the source file to `raw_input/<year>/`.

No LLM involved. Pure file parsing.

---

## Setup

```bash
pip install -r requirements.txt
```

Tesseract OCR engine (required for scanned PDFs and images):

```bash
# Mac
brew install tesseract

# Ubuntu / Debian
sudo apt install tesseract-ocr

# Windows
# https://github.com/UB-Mannheim/tesseract/wiki
```

For Hindi text in reports:
```bash
brew install tesseract-lang          # Mac
sudo apt install tesseract-ocr-hin   # Ubuntu
```

### First run — folder setup

Set `BASE_DIR` in the `Config` class to your project folder, then run:

```bash
python extractor.py setup
```

This creates the full folder structure under `BASE_DIR` in one shot:

```
<BASE_DIR>/
├── raw_input/
│   └── incoming/        ← drop files here
├── raw_extracted/
│   └── incoming/        ← extracted .md files appear here
└── wiki/
    ├── distillations/   ← wiki/distillation .md files land here
    ├── concepts/
    ├── people/
    ├── summaries/
    ├── traditions/
    ├── debates/
    └── organizations/
```

Safe to run on an existing project — folders that already exist are
left untouched. `setup` also runs automatically at the start of every
`watch`, `batch`, and `file` command, so you never have to remember
to run it separately.

---

## Usage

### First time — create folder structure
```bash
python extractor.py setup
```

### Watch mode (recommended — runs continuously)
```bash
python extractor.py
```
Drop any file into `raw_input/incoming/` — it processes automatically.

### Single file
```bash
python extractor.py file path/to/report.pdf
```

### Batch mode (first-time bulk processing)
```bash
# Process everything in raw_input/incoming/
python extractor.py batch

# Process a specific folder (e.g. your 2019 archive)
python extractor.py batch path/to/raw_input/2019/
```

---

## Supported file types

| Extension | Extractor | Notes |
|---|---|---|
| .pdf | pdfplumber → OCR fallback | Auto-detects scanned vs text |
| .xlsx .xls .xlsm | pandas | All sheets extracted as markdown tables |
| .docx .doc | python-docx | Headings and tables preserved |
| .jpg .jpeg .png | tesseract or easyocr | Set OCR_ENGINE in Config |
| .json | built-in | Pretty-printed |
| .txt .csv | built-in | Pass-through |
| .md | smart router | See below |

### .md file routing

`.md` files are not all the same — the extractor detects what kind
they are and routes accordingly:

| What it detects | Example | Outcome |
|---|---|---|
| Contains `[[wikilinks]]` | A wiki concept page | Moved to `wiki/distillations/` — no extraction needed |
| Has Obsidian frontmatter (`tags:` / `type:`) | A distillation you wrote | Moved to `wiki/distillations/` — already final form |
| Has extractor frontmatter (`source_file:`) | Previously processed file | Archived to `raw_input/<year>/` — not reprocessed |
| Plain prose, none of the above | Typed clinical note | Normal pipeline — classified and sent to `raw_extracted/incoming/` |

This means you can drop wiki pages, distillations, and clinical notes
all into `incoming/` — each goes to the right place automatically.

---

## Output format

Each extracted file has a frontmatter header:

```markdown
---
source_file: ajit-lipid-2024.pdf
source_path: /path/to/raw_input/incoming/ajit-lipid-2024.pdf
extracted_date: 2026-05-06
content_type: lab-report
person: ajit
year: 2024
confidence: high
slug: ajit-lipid-2024
---
```

`content_type` is used by Service 2 (Synthesizer) to select the
correct wiki template.

Confidence levels:
- `high` — text PDF or xlsx, clean extraction
- `medium` — OCR used, likely good but may have errors
- `low` — extraction failed or sparse, needs manual review

---

## Customising the classifier

Edit `CONTENT_TYPE_RULES` and `PERSON_RULES` in `extractor.py`
to add new patterns:

```python
CONTENT_TYPE_RULES = [
    (r"lab|blood|cbc", "lab-report"),
    (r"your-new-pattern", "your-new-type"),
    ...
]

PERSON_RULES = [
    (r"\bajit\b", "ajit"),
    (r"\basha\b", "asha"),
    (r"\bnew-person\b", "new-person"),
]
```

---

## What this service does NOT do

- No LLM calls
- No wiki writes
- No interpretation of clinical values
- No modification of files in `raw_input/` (only moves from incoming/)

Service 2 (Synthesizer) handles everything after this point.
