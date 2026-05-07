"""
Service 1 — Extractor
Watches raw_input/incoming/ for new files.
Extracts content to structured markdown in raw_extracted/incoming/.
Moves source file to raw_input/<year>/.
No LLM involved.
"""

import os
import re
import shutil
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("extractor")


# ── Config ───────────────────────────────────────────────────────────────────

class Config:
    """
    Reads BASE_DIR from .env in the service root.
    All data folders are resolved relative to BASE_DIR so both services
    point to the same project directory.
    """
    # Load .env from service root (one level up from src/)
    load_dotenv(Path(__file__).parent.parent / ".env")

    BASE_DIR               = Path(os.getenv("BASE_DIR", str(Path(__file__).parent.parent)))
    RAW_INPUT_INCOMING     = BASE_DIR / "raw_input" / "incoming"
    RAW_EXTRACTED_INCOMING = BASE_DIR / "raw_extracted" / "incoming"
    RAW_INPUT_ARCHIVE      = BASE_DIR / "raw_input"
    RAW_EXTRACTED_ARCHIVE  = BASE_DIR / "raw_extracted"

    # OCR settings — override in .env if needed
    TESSERACT_LANG = os.getenv("TESSERACT_LANG", "eng")
    OCR_ENGINE     = os.getenv("OCR_ENGINE", "tesseract")

    # Supported extensions
    SUPPORTED_EXTS = {
        ".pdf", ".xlsx", ".xls", ".xlsm",
        ".docx", ".doc",
        ".jpg", ".jpeg", ".png",
        ".json", ".md", ".txt", ".csv",
    }


# ── Content type classifier ──────────────────────────────────────────────────

CONTENT_TYPE_RULES = [
    # Rules are checked in order — first match wins.
    # More specific patterns come before general ones.
    # Patterns are matched against filename + first 1000 chars of text.

    # Lab reports — comprehensive biochemistry/haematology panels
    (r"haemogram|hemogram|biochem|lft|kft|hba1c|glycosylated|lipid.profile|"
     r"complete.blood|cbc|haematology|liver.function|renal.profile|"
     r"thyroid.function|blood.glucose|serum.creatinine", "lab-report"),

    # Imaging — specific modalities (before generic lab-report catch-all)
    (r"echo|echocardiogram|echocardiography", "imaging-report-echo"),
    (r"cac|calcium.score|coronary.artery.calcium|agatston", "imaging-report-cac"),
    (r"usg|ultrasound|sonograph|sonography", "imaging-report-usg"),
    # ct/mri must be whole words or with space — avoid matching "oct", "doctor" etc.
    (r"\bct\b|\bmri\b|x-ray|xray|\bchest\b|radiograph", "imaging-report-radiology"),

    # Smart/AI reports
    (r"smart.?report|ai.?report|health.?report|summary.?report", "smart-report"),

    # Trends / historical data
    (r"trend|tracker|history|monitoring", "trend-data"),

    # Clinical notes / prescriptions
    (r"prescription|\brx\b|medicine|medication|discharge.summary", "clinical-note"),

    # Generic lab catch-all — after all specific patterns
    (r"lab|blood|serum|plasma|urine|culture|thyroid", "lab-report"),

    # Billing — non-clinical
    (r"invoice|bill|receipt", "billing"),
]

PERSON_RULES = [
    (r"\bperson_1\b", "person_1"),
    (r"\bperson_2\b", "person_2"),
]


def classify_content_type(filename: str, text_sample: str) -> str:
    combined = (filename + " " + text_sample[:1000]).lower()
    for pattern, ctype in CONTENT_TYPE_RULES:
        if re.search(pattern, combined):
            return ctype
    return "unknown"


def classify_person(filename: str, text_sample: str) -> str:
    combined = (filename + " " + text_sample[:500]).lower()
    for pattern, person in PERSON_RULES:
        if re.search(pattern, combined):
            return person
    return "unknown"


def infer_year(filepath: Path, text_sample: str) -> str:
    # 1. Year from parent folder name
    for part in filepath.parts:
        if re.match(r"^20\d{2}$", part):
            return part
    # 2. Year from filename
    m = re.search(r"(20\d{2})", filepath.name)
    if m:
        return m.group(1)
    # 3. Year from text (first 4-digit year found)
    m = re.search(r"(20\d{2})", text_sample[:1000])
    if m:
        return m.group(1)
    # 4. Default to current year
    return str(datetime.now().year)


# ── Extractors ───────────────────────────────────────────────────────────────

def extract_pdf(filepath: Path) -> tuple[str, str]:
    """
    Returns (extracted_text, confidence).
    Tries text extraction first; falls back to OCR if text is sparse.
    """
    import pdfplumber

    text_pages = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                # Extract text
                page_text = page.extract_text() or ""
                # Extract tables as markdown
                tables = page.extract_tables()
                table_md = ""
                for table in tables:
                    if not table:
                        continue
                    rows = []
                    for i, row in enumerate(table):
                        clean_row = [str(c or "").strip() for c in row]
                        rows.append("| " + " | ".join(clean_row) + " |")
                        if i == 0:
                            rows.append("|" + "|".join(["---"] * len(row)) + "|")
                    table_md += "\n".join(rows) + "\n\n"
                text_pages.append(page_text + "\n" + table_md)

        full_text = "\n\n---\n\n".join(text_pages).strip()

        # If text is too sparse, it's probably a scanned PDF
        word_count = len(full_text.split())
        if word_count < 30:
            log.info(f"  Sparse text ({word_count} words) — falling back to OCR")
            return extract_pdf_ocr(filepath)

        return full_text, "high"

    except Exception as e:
        log.warning(f"  pdfplumber failed: {e} — trying OCR")
        return extract_pdf_ocr(filepath)


def extract_pdf_ocr(filepath: Path) -> tuple[str, str]:
    """OCR fallback for scanned PDFs."""
    try:
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image
        import io

        doc = fitz.open(str(filepath))
        pages_text = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Render at 200 DPI for decent OCR quality
            mat = fitz.Matrix(200 / 72, 200 / 72)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            text = pytesseract.image_to_string(
                img, lang=Config.TESSERACT_LANG
            )
            pages_text.append(text)

        return "\n\n---\n\n".join(pages_text).strip(), "medium"

    except Exception as e:
        log.error(f"  OCR also failed: {e}")
        return "", "low"


def extract_xlsx(filepath: Path) -> tuple[str, str]:
    """Extract xlsx/xls to markdown tables."""
    import pandas as pd

    try:
        xl = pd.ExcelFile(filepath)
        sheets_md = []
        for sheet_name in xl.sheet_names:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
            if df.empty:
                continue
            md = f"## Sheet: {sheet_name}\n\n"
            md += df.fillna("").to_markdown(index=False)
            sheets_md.append(md)
        return "\n\n".join(sheets_md), "high"
    except Exception as e:
        log.error(f"  xlsx extraction failed: {e}")
        return "", "low"


def extract_docx(filepath: Path) -> tuple[str, str]:
    """Extract docx to markdown."""
    try:
        import docx
        doc = docx.Document(str(filepath))
        parts = []
        for para in doc.paragraphs:
            if para.text.strip():
                # Preserve heading levels
                if para.style.name.startswith("Heading"):
                    level = para.style.name.split()[-1]
                    try:
                        hashes = "#" * int(level)
                    except ValueError:
                        hashes = "##"
                    parts.append(f"{hashes} {para.text}")
                else:
                    parts.append(para.text)
        # Extract tables
        for table in doc.tables:
            rows = []
            for i, row in enumerate(table.rows):
                cells = [c.text.strip() for c in row.cells]
                rows.append("| " + " | ".join(cells) + " |")
                if i == 0:
                    rows.append("|" + "|".join(["---"] * len(cells)) + "|")
            parts.append("\n".join(rows))
        return "\n\n".join(parts), "high"
    except Exception as e:
        log.error(f"  docx extraction failed: {e}")
        return "", "low"


def extract_image(filepath: Path) -> tuple[str, str]:
    """OCR an image file."""
    try:
        if Config.OCR_ENGINE == "easyocr":
            import easyocr
            reader = easyocr.Reader(["en"])
            results = reader.readtext(str(filepath), detail=0)
            return "\n".join(results), "medium"
        else:
            import pytesseract
            from PIL import Image
            img = Image.open(filepath)
            text = pytesseract.image_to_string(img, lang=Config.TESSERACT_LANG)
            return text.strip(), "medium"
    except Exception as e:
        log.error(f"  image OCR failed: {e}")
        return "", "low"


def extract_json(filepath: Path) -> tuple[str, str]:
    try:
        with open(filepath) as f:
            data = json.load(f)
        return json.dumps(data, indent=2), "high"
    except Exception as e:
        log.error(f"  json read failed: {e}")
        return "", "low"


def extract_text(filepath: Path) -> tuple[str, str]:
    try:
        return filepath.read_text(encoding="utf-8", errors="replace"), "high"
    except Exception as e:
        log.error(f"  text read failed: {e}")
        return "", "low"


def extract_md(filepath: Path) -> tuple[str, str, str]:
    """
    Special handler for .md files. Returns (text, confidence, disposition).

    disposition values:
      "process"   — plain prose; treat as source document, run normal pipeline
      "skip"      — wiki/distillation page with wikilinks or Obsidian frontmatter
      "duplicate" — already has extractor frontmatter; processed before
    """
    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        log.error(f"  md read failed: {e}")
        return "", "low", "process"

    # Already has extractor frontmatter — processed before
    if "source_file:" in content and "content_type:" in content:
        log.info("  md has extractor frontmatter — duplicate, skipping")
        return content, "high", "duplicate"

    # Obsidian-style frontmatter with tags — wiki or distillation page
    if content.startswith("---") and ("tags:" in content or "type:" in content):
        log.info("  md looks like a wiki/distillation page — skipping pipeline")
        return content, "high", "skip"

    # Contains wikilinks — it's a wiki page
    if "[[" in content and "]]" in content:
        log.info("  md contains wikilinks — treating as wiki page, skipping")
        return content, "high", "skip"

    # Plain prose — treat as typed clinical note or source document
    log.info("  md is plain prose — treating as source document")
    return content, "high", "process"


# ── Router ───────────────────────────────────────────────────────────────────

def extract(filepath: Path) -> tuple[str, str]:
    """Route file to correct extractor. Returns (text, confidence)."""
    ext = filepath.suffix.lower()
    if ext == ".pdf":
        return extract_pdf(filepath)
    elif ext in {".xlsx", ".xls", ".xlsm"}:
        return extract_xlsx(filepath)
    elif ext in {".docx", ".doc"}:
        return extract_docx(filepath)
    elif ext in {".jpg", ".jpeg", ".png"}:
        return extract_image(filepath)
    elif ext == ".json":
        return extract_json(filepath)
    elif ext in {".txt", ".csv"}:
        return extract_text(filepath)
    elif ext == ".md":
        # md has its own disposition logic — handled in process_file
        text, confidence, _ = extract_md(filepath)
        return text, confidence
    else:
        log.warning(f"  Unsupported extension: {ext}")
        return "", "low"


# ── Output writer ─────────────────────────────────────────────────────────────

def write_extracted_md(
    source_path: Path,
    extracted_text: str,
    confidence: str,
    content_type: str,
    person: str,
    year: str,
    out_path: Path,
) -> None:
    """Write the structured extracted markdown file."""
    now = datetime.now().strftime("%Y-%m-%d")
    slug = source_path.stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")

    frontmatter = f"""---
source_file: {source_path.name}
source_path: {source_path}
extracted_date: {now}
content_type: {content_type}
person: {person}
year: {year}
confidence: {confidence}
slug: {slug}
---
"""

    body = f"""# Extracted: {source_path.name}

{extracted_text}
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(frontmatter + body, encoding="utf-8")
    log.info(f"  Written → {out_path}")


# ── Archive mover ─────────────────────────────────────────────────────────────

def archive_source(source_path: Path, year: str) -> Path:
    """Move source file from incoming/ to raw_input/<year>/."""
    dest_dir = Config.RAW_INPUT_ARCHIVE / year
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / source_path.name
    # Avoid overwriting if same file already archived
    if dest.exists():
        stem = source_path.stem
        suffix = source_path.suffix
        dest = dest_dir / f"{stem}_{datetime.now().strftime('%H%M%S')}{suffix}"
    shutil.move(str(source_path), str(dest))
    log.info(f"  Archived → {dest}")
    return dest


# ── Core process function ─────────────────────────────────────────────────────

def process_file(filepath: Path) -> bool:
    """
    Full pipeline for one file.
    Returns True on success, False on failure.
    """
    log.info(f"Processing: {filepath.name}")

    if filepath.suffix.lower() not in Config.SUPPORTED_EXTS:
        log.warning(f"  Skipping unsupported type: {filepath.suffix}")
        return False

    # ── Special handling for .md files ───────────────────────────────────────
    if filepath.suffix.lower() == ".md":
        text, confidence, disposition = extract_md(filepath)

        if disposition == "duplicate":
            log.info(f"  Duplicate — removing from incoming without reprocessing")
            year = infer_year(filepath, text)
            archive_source(filepath, year)
            return True

        if disposition == "skip":
            # It's a wiki/distillation page — move to wiki/distillations/
            # (Service 2 does not need to process it; it already is a wiki page)
            dest_dir = Config.BASE_DIR / "wiki" / "distillations"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / filepath.name
            if dest.exists():
                log.warning(f"  {filepath.name} already exists in wiki/distillations/ — skipping move")
            else:
                shutil.move(str(filepath), str(dest))
                log.info(f"  Wiki/distillation page moved → {dest}")
            return True

        # disposition == "process" — plain prose md, treat as source document
        # Fall through to normal pipeline below
        extracted_text = text

    else:
        # 1. Extract
        extracted_text, confidence = extract(filepath)

    if not extracted_text.strip():
        log.error(f"  No content extracted from {filepath.name}")
        return False

    # 2. Classify
    content_type = classify_content_type(filepath.name, extracted_text)
    person       = classify_person(filepath.name, extracted_text)
    year         = infer_year(filepath, extracted_text)

    log.info(f"  type={content_type}  person={person}  year={year}  confidence={confidence}")

    # 3. Write extracted md
    out_filename = filepath.stem + ".md"
    out_path = Config.RAW_EXTRACTED_INCOMING / out_filename
    write_extracted_md(
        source_path=filepath,
        extracted_text=extracted_text,
        confidence=confidence,
        content_type=content_type,
        person=person,
        year=year,
        out_path=out_path,
    )

    # 4. Archive source file
    archive_source(filepath, year)

    log.info(f"  Done: {filepath.name}")
    return True


# ── Watcher ───────────────────────────────────────────────────────────────────

def run_watcher():
    """Watch raw_input/incoming/ and process new files."""
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    class Handler(FileSystemEventHandler):
        def on_created(self, event):
            if event.is_directory:
                return
            filepath = Path(event.src_path)
            # Brief delay to ensure file is fully written
            import time; time.sleep(1)
            if filepath.exists():
                process_file(filepath)

    watch_dir = Config.RAW_INPUT_INCOMING

    observer = Observer()
    observer.schedule(Handler(), str(watch_dir), recursive=False)
    observer.start()
    log.info(f"Watching: {watch_dir}")
    log.info("Drop files into raw_input/incoming/ to process them.")
    log.info("Ctrl+C to stop.\n")

    # Process any files already present before the watcher started
    existing = [
        f for f in sorted(watch_dir.iterdir())
        if f.is_file() and f.suffix.lower() in Config.SUPPORTED_EXTS
    ] if watch_dir.exists() else []
    if existing:
        log.info(f"Found {len(existing)} existing file(s) in incoming/ — processing now")
        for f in existing:
            process_file(f)

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


# ── Batch mode (process existing files) ──────────────────────────────────────

def run_batch(folder: Optional[Path] = None):
    """
    Process all supported files in a folder (default: raw_input/incoming/).
    Useful for first-time bulk ingestion of an existing raw_input/ tree.
    """
    target = folder or Config.RAW_INPUT_INCOMING
    files = [
        f for f in target.rglob("*")
        if f.is_file() and f.suffix.lower() in Config.SUPPORTED_EXTS
    ]
    log.info(f"Batch mode: {len(files)} files found in {target}")
    success, fail = 0, 0
    for f in sorted(files):
        ok = process_file(f)
        if ok: success += 1
        else:  fail    += 1
    log.info(f"Batch complete: {success} succeeded, {fail} failed")


# ── First-run setup ───────────────────────────────────────────────────────────

REQUIRED_DIRS = [
    # Extractor pipeline folders
    "raw_input/incoming",
    "raw_extracted/incoming",

    # Health wiki folders
    "wiki/people",          # master summary per person
    "wiki/reports",         # one md per source document
    "wiki/biomarkers",      # per-parameter time-series tables
    "wiki/conditions",      # per diagnosed/monitored condition
    "wiki/medications",     # current and past medications
    "wiki/imaging",         # USG, ECHO, CAC, X-ray, CT findings
    "wiki/trends",          # per-person narrative trajectory
    "wiki/correlations",    # inter-parameter relationships (LLM-discovered)
    "wiki/insights",        # your own notes and next steps
]

def setup():
    """
    Create all required folders under BASE_DIR if they don't exist.
    Safe to run multiple times — existing folders are untouched.
    Prints a summary of what was created vs already present.
    """
    created = []
    existed = []

    for rel in REQUIRED_DIRS:
        d = Config.BASE_DIR / rel
        if d.exists():
            existed.append(rel)
        else:
            d.mkdir(parents=True, exist_ok=True)
            created.append(rel)

    if created:
        log.info("Setup — created folders:")
        for d in created:
            log.info(f"  + {d}")
    if existed:
        log.info("Setup — already present:")
        for d in existed:
            log.info(f"  ✓ {d}")

    log.info(f"Setup complete. Base directory: {Config.BASE_DIR}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Always run setup first — idempotent, safe every time
    setup()

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "setup":
            # python extractor.py setup — just create folders and exit
            pass
        elif cmd == "batch":
            # python extractor.py batch [optional/path/to/folder]
            folder = Path(sys.argv[2]) if len(sys.argv) > 2 else None
            run_batch(folder)
        elif cmd == "file":
            # python extractor.py file path/to/single/file.pdf
            if len(sys.argv) < 3:
                print("Usage: python extractor.py file <filepath>")
                sys.exit(1)
            process_file(Path(sys.argv[2]))
        else:
            print(f"Unknown command: {cmd}")
            print("Commands: setup | watch | batch [folder] | file <filepath>")
            sys.exit(1)
    else:
        # Default: watch mode
        run_watcher()
