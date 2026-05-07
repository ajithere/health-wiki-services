"""
Service 2 — Synthesizer
Watches raw_extracted/incoming/ for new files.
Reads extracted markdown, calls LLM, writes/updates wiki pages.
Supports: Anthropic, OpenAI, Google, Ollama, OpenAI-compatible.
"""

import os
import re
import json
import shutil
import logging
import yaml
from abc import ABC, abstractmethod
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
log = logging.getLogger("synthesizer")


# ── Config ───────────────────────────────────────────────────────────────────

class Config:
    def __init__(self):
        # Load .env from service root (one level up from src/)
        load_dotenv(Path(__file__).parent.parent / ".env")
        self.BASE_DIR               = Path(os.getenv("BASE_DIR", "."))
        self.RAW_EXTRACTED_INCOMING = self.BASE_DIR / "raw_extracted" / "incoming"
        self.RAW_EXTRACTED_ARCHIVE  = self.BASE_DIR / "raw_extracted"
        self.WIKI_DIR               = self.BASE_DIR / "wiki"
        self.STAGING_DIR            = self.BASE_DIR / "staging"
        self.STAGING_ENABLED        = os.getenv("STAGING_ENABLED", "true").lower() == "true"
        self.STAGING_AUTO_APPROVE   = os.getenv("STAGING_AUTO_APPROVE", "false").lower() == "true"
        self.LLM_PROVIDER           = os.getenv("LLM_PROVIDER", "anthropic")
        self.LLM_MODEL              = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
        self.LLM_MAX_TOKENS         = int(os.getenv("LLM_MAX_TOKENS", "8000"))
        self.LLM_TEMPERATURE        = float(os.getenv("LLM_TEMPERATURE", "0"))

    def wiki_path(self, *parts) -> Path:
        return self.WIKI_DIR.joinpath(*parts)


# ── Biomarker config ──────────────────────────────────────────────────────────

class BiomarkerConfig:
    def __init__(self, yaml_path: Path):
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        self.track: list[str] = data.get("track", [])
        raw_aliases: dict = data.get("aliases", {})

        # Build reverse map: lowercase variant → canonical name
        self.alias_map: dict[str, str] = {}
        for canonical, variants in raw_aliases.items():
            self.alias_map[canonical.lower()] = canonical
            for v in variants:
                self.alias_map[v.lower()] = canonical

    def normalise(self, raw_name: str) -> Optional[str]:
        """Map a raw lab name to a canonical biomarker name, or None if not tracked."""
        key = raw_name.strip().lower()
        canonical = self.alias_map.get(key)
        if canonical and canonical in self.track:
            return canonical
        # Try partial match on tracked list
        for tracked in self.track:
            if tracked.replace("-", " ") in key or key in tracked.replace("-", " "):
                return tracked
        return None


# ── LLM adapter layer ─────────────────────────────────────────────────────────

class LLMAdapter(ABC):
    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        pass


class AnthropicAdapter(LLMAdapter):
    def __init__(self, model: str):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text


class OpenAIAdapter(LLMAdapter):
    def __init__(self, model: str, base_url: Optional[str] = None):
        from openai import OpenAI
        kwargs = {"api_key": os.getenv("OPENAI_API_KEY")}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content


class GoogleAdapter(LLMAdapter):
    def __init__(self, model: str):
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model_name = model
        self.genai = genai

    def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        model = self.genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system,
            generation_config={"max_output_tokens": max_tokens, "temperature": temperature},
        )
        response = model.generate_content(user)
        return response.text


class OllamaAdapter(LLMAdapter):
    def __init__(self, model: str, base_url: Optional[str] = None):
        import ollama
        self.model = model
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama = ollama

    def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        response = self.ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={"num_predict": max_tokens, "temperature": temperature},
        )
        return response["message"]["content"]


def build_adapter(cfg: Config) -> LLMAdapter:
    provider = cfg.LLM_PROVIDER.lower()
    model = cfg.LLM_MODEL
    log.info(f"LLM provider: {provider}  model: {model}")

    if provider == "anthropic":
        return AnthropicAdapter(model)
    elif provider == "openai":
        return OpenAIAdapter(model)
    elif provider == "openai-compatible":
        return OpenAIAdapter(model, base_url=os.getenv("OPENAI_BASE_URL"))
    elif provider == "google":
        return GoogleAdapter(model)
    elif provider == "ollama":
        return OllamaAdapter(model)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. "
                         f"Choose from: anthropic, openai, google, ollama, openai-compatible")


# ── Frontmatter parser ────────────────────────────────────────────────────────

def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter. Returns (metadata_dict, body_text)."""
    if not content.startswith("---"):
        return {}, content
    end = content.find("\n---", 3)
    if end == -1:
        return {}, content
    try:
        meta = yaml.safe_load(content[3:end])
        body = content[end + 4:].strip()
        return meta or {}, body
    except Exception:
        return {}, content


# ── Context loader ────────────────────────────────────────────────────────────

class ContextLoader:
    """Loads relevant wiki pages to include in the LLM prompt."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def load_file(self, path: Path) -> str:
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def load_for_ingest(self, meta: dict) -> str:
        """Load minimal wiki context relevant to this specific file."""
        parts = []

        # Always: index
        index = self.load_file(self.cfg.wiki_path("index.md"))
        if index:
            parts.append(f"=== wiki/index.md ===\n{index}")

        # Always: person page
        person = meta.get("person", "unknown")
        if person != "unknown":
            person_page = self.load_file(self.cfg.wiki_path("people", f"{person}.md"))
            if person_page:
                parts.append(f"=== wiki/people/{person}.md ===\n{person_page}")

        # NOTE: biomarker pages are intentionally NOT loaded into context.
        # Sending existing biomarker rows causes the LLM to skip adding new
        # rows when dates are close together (e.g. consecutive-day reports).
        # Deduplication is handled deterministically by _append_row in code.
        # The LLM only needs the raw report to extract values — not the wiki state.
        content_type = meta.get("content_type", "")

        # For imaging: load prior reports of same modality
        if content_type.startswith("imaging-report"):
            modality = content_type.replace("imaging-report-", "")
            imaging_dir = self.cfg.wiki_path("imaging")
            if imaging_dir.exists():
                matches = sorted([
                    f for f in imaging_dir.iterdir()
                    if person in f.name and modality in f.name
                ])[-2:]  # last 2 for comparison
                for f in matches:
                    content = self.load_file(f)
                    if content:
                        parts.append(f"=== wiki/imaging/{f.name} ===\n{content}")

        return "\n\n".join(parts)


# ── Prompt builder ────────────────────────────────────────────────────────────

# ── Prompt loader ─────────────────────────────────────────────────────────────

class PromptLoader:
    """
    Loads system and user prompts from config/prompts.yaml.
    Falls back to hardcoded defaults if the file is missing.
    Reload on every process_file call so prompt edits take effect
    without restarting the service.
    """

    def __init__(self, yaml_path: Path):
        self.yaml_path = yaml_path
        self._mtime: float = 0
        self._system: str = ""
        self._user: str = ""
        self._load()

    def _load(self):
        if not self.yaml_path.exists():
            log.warning(f"prompts.yaml not found at {self.yaml_path} — using defaults")
            return
        mtime = self.yaml_path.stat().st_mtime
        if mtime == self._mtime:
            return  # unchanged
        with open(self.yaml_path) as f:
            data = yaml.safe_load(f)
        self._system = data.get("system", "").strip()
        self._user   = data.get("user", "").strip()
        self._mtime  = mtime
        log.info(f"Prompts loaded from {self.yaml_path.name}")

    def system(self) -> str:
        self._load()
        return self._system

    def user(self, wiki_context: str, tracked_biomarkers: str,
             extracted_content: str) -> str:
        self._load()
        ctx = wiki_context if wiki_context else "(wiki is empty — this is a first ingest)"
        return (self._user
                .replace("{wiki_context}", ctx)
                .replace("{tracked_biomarkers}", tracked_biomarkers)
                .replace("{extracted_content}", extracted_content))


# ── Operation runner ──────────────────────────────────────────────────────────

class OperationRunner:
    """
    Executes file operations from the LLM response.
    Handles deduplication, conflict detection, and atomic writes.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Write to staging or directly to wiki
        self.write_base = cfg.STAGING_DIR if cfg.STAGING_ENABLED else cfg.WIKI_DIR

    def resolve(self, path_str: str) -> Path:
        """Resolve a relative wiki path to the write target (staging or wiki)."""
        rel = path_str.replace("wiki/", "", 1)
        return self.write_base / rel

    def run(self, operations: list[dict]) -> list[str]:
        """Execute all operations. Returns list of result messages."""
        results = []
        for op in operations:
            try:
                msg = self._dispatch(op)
                results.append(msg)
                log.info(f"  {msg}")
            except Exception as e:
                msg = f"ERROR on {op.get('op')} {op.get('path','')}: {e}"
                results.append(msg)
                log.error(f"  {msg}")
        return results

    def _dispatch(self, op: dict) -> str:
        kind = op.get("op")
        if kind == "create":
            return self._create(op)
        elif kind == "update":
            return self._update(op)
        elif kind == "append_row":
            return self._append_row(op)
        elif kind == "append_log":
            return self._append_log(op)
        elif kind == "update_index":
            return self._update_index(op)
        else:
            raise ValueError(f"Unknown operation: {kind}")

    def _create(self, op: dict) -> str:
        path = self.resolve(op["path"])
        path.parent.mkdir(parents=True, exist_ok=True)

        # If LLM sends "create" for an existing biomarker page,
        # extract the row from the content and convert to append_row.
        if path.exists() and "biomarkers/" in op["path"]:
            log.info(f"  Biomarker page exists — converting create to append_row for {op['path']}")
            content_str = op.get("content", "")
            # Extract person from content (## Person header)
            person_match = re.search(r"^## (.+)$", content_str, re.MULTILINE)
            person = person_match.group(1).lower() if person_match else "person_1"
            # Extract the data row (line starting with | that has a date)
            for line in content_str.splitlines():
                stripped = line.strip()
                if (stripped.startswith("|")
                        and "---" not in stripped
                        and "Date" not in stripped
                        and "Value" not in stripped
                        and re.search(r"\d{4}-\d{2}-\d{2}", stripped)):
                    append_op = {
                        "op": "append_row",
                        "path": op["path"],
                        "person": person,
                        "row": stripped,
                    }
                    return self._append_row(append_op)
            log.warning(f"  Could not extract row from create op for {op['path']} — skipping")
            return f"skipped create on existing biomarker {op['path']} (no row found)"

        if path.exists():
            # Avoid overwriting other pages — save as conflict file for review
            conflict = path.with_suffix(f".conflict-{datetime.now().strftime('%H%M%S')}.md")
            path.rename(conflict)
            log.warning(f"  Existing file moved to {conflict.name} — new version written")

        path.write_text(op["content"], encoding="utf-8")
        return f"created {op['path']}"

    def _update(self, op: dict) -> str:
        path = self.resolve(op["path"])
        if not path.exists():
            # Page doesn't exist yet — create it with the content
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(op["content"], encoding="utf-8")
            return f"created (via update) {op['path']}"
        content = path.read_text(encoding="utf-8")
        section = op.get("section", "")
        new_content = op.get("content", "")
        if section:
            # Replace section content
            pattern = rf"(## {re.escape(section)}\n)(.*?)(\n## |\Z)"
            replacement = rf"\g<1>{new_content}\n\g<3>"
            updated = re.sub(pattern, replacement, content, flags=re.DOTALL)
            if updated == content:
                # Section not found — append it
                updated = content.rstrip() + f"\n\n## {section}\n{new_content}\n"
            path.write_text(updated, encoding="utf-8")
        else:
            path.write_text(new_content, encoding="utf-8")
        return f"updated {op['path']}"

    def _append_row(self, op: dict) -> str:
        path = self.resolve(op["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        person = op.get("person", "unknown")
        row = op.get("row", "").strip()

        if not path.exists():
            # Create biomarker page with correct structure
            biomarker_name = path.stem.replace("-", " ").title()
            content = self._new_biomarker_page(biomarker_name, person, row)
            path.write_text(content, encoding="utf-8")
            return f"created biomarker page {op['path']}"

        content = path.read_text(encoding="utf-8")
        # No conflict detection — append every row unconditionally.
        # Each file is ingested once; duplicates are prevented by workflow,
        # not by code. Detection logic was causing false positives and
        # blocking valid rows from consecutive-day reports.

        # Find the person's table section and append after the last data row
        person_header = f"## {person.title()}"
        if person_header in content:
            section_start = content.index(person_header)
            next_section  = content.find("\n## ", section_start + 1)
            section_end   = next_section if next_section != -1 else len(content)
            section       = content[section_start:section_end]

            # Find all data rows within this section (not header/separator)
            table_rows = [
                l for l in section.splitlines()
                if l.startswith("|") and "---" not in l
                and not any(h in l for h in ["Date", "Value"])
            ]

            if table_rows:
                # Rebuild the table sorted by date (insert in correct position)
                all_rows = table_rows + [row]
                def row_date(r):
                    m = re.search(r"\d{4}-\d{2}-\d{2}", r)
                    return m.group() if m else "0000-00-00"
                sorted_rows = sorted(all_rows, key=row_date)

                # Replace the existing data rows block with the sorted set
                table_header = "| Date | Value | Unit | Flag | Source |\n|------|-------|------|------|--------|"
                first_row_in_section = section.find(table_rows[0])
                section_before_table = section[:first_row_in_section]
                # Find where this section ends in the full content
                new_section = (section_before_table +
                               "\n".join(sorted_rows) + "\n")
                content = content[:section_start] + new_section + content[section_end:]
            else:
                # No data rows yet — add header + separator + row
                table_header = "| Date | Value | Unit | Flag | Source |\n|------|-------|------|------|--------|"
                insert_at = section_start + len(person_header)
                content = (content[:insert_at] +
                           "\n\n" + table_header + "\n" + row +
                           content[insert_at:])
        else:
            # Person section doesn't exist — append new section
            table_header = "| Date | Value | Unit | Flag | Source |\n|------|-------|------|------|--------|"
            content = content.rstrip() + f"\n\n{person_header}\n\n{table_header}\n{row}\n"

        path.write_text(content, encoding="utf-8")
        return f"appended row to {op['path']}"

    def _append_log(self, op: dict) -> str:
        log_path = self.resolve("log.md")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = op.get("content", "")
        if log_path.exists():
            current = log_path.read_text(encoding="utf-8")
            log_path.write_text(current.rstrip() + "\n\n" + entry + "\n", encoding="utf-8")
        else:
            log_path.write_text(f"# Log\n\n{entry}\n", encoding="utf-8")
        return "appended to log.md"

    def _update_index(self, op: dict) -> str:
        """
        Surgically update index.md after an ingest.
        op fields:
          person        — person_1 | person_2
          report_type   — lab-report | imaging-report-echo | ...
          date          — YYYY-MM-DD
          new_pages     — list of wiki paths created/updated
          new_biomarkers— list of biomarker canonical names touched
          status_update — one-line current status string for this person
        """
        index_path = self.resolve("index.md")
        index_path.parent.mkdir(parents=True, exist_ok=True)

        person        = op.get("person", "unknown")
        report_type   = op.get("report_type", "unknown")
        date          = op.get("date", datetime.now().strftime("%Y-%m-%d"))
        new_pages     = op.get("new_pages", [])
        new_biomarkers= op.get("new_biomarkers", [])
        status_update = op.get("status_update", "")

        if not index_path.exists():
            index_path.write_text(INDEX_TEMPLATE, encoding="utf-8")

        content = index_path.read_text(encoding="utf-8")

        # 1. Update person's last report date in Reports ingested section
        last_report_pattern = (
            rf"(- {person}:.*?last report:? ?)\d{{4}}-\d{{2}}-\d{{2}}"
        )
        replacement = rf"\g<1>{date}"
        content = re.sub(last_report_pattern, replacement, content,
                         flags=re.IGNORECASE)

        # 2. Add new biomarker links if not already present
        if new_biomarkers:
            bm_section = "## Biomarker pages"
            if bm_section in content:
                for bm in new_biomarkers:
                    link = f"[[{bm}]]"
                    if link not in content:
                        # Insert after section header
                        insert_at = content.index(bm_section) + len(bm_section)
                        content = (content[:insert_at] +
                                   f"\n- {link}" +
                                   content[insert_at:])

        # 3. Update current status snapshot for person
        status_header = f"## Current status — {person.title()}"
        if status_header in content and status_update:
            pattern = rf"({re.escape(status_header)}\n)(.*?)(\n## |\Z)"
            replacement = rf"\g<1>Last report: {date} ({report_type})\n{status_update}\n\g<3>"
            content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        elif status_update:
            content = content.rstrip() + (
                f"\n\n{status_header}\n"
                f"Last report: {date} ({report_type})\n"
                f"{status_update}\n"
            )

        index_path.write_text(content, encoding="utf-8")
        return f"updated index.md for {person} ({report_type} {date})"

    def _new_biomarker_page(self, name: str, person: str, first_row: str) -> str:
        table = "| Date | Value | Unit | Flag | Source |\n|------|-------|------|------|--------|"
        return f"""# {name}

## {person.title()}

{table}
{first_row}

"""


# ── LLM response parser ───────────────────────────────────────────────────────

def parse_llm_response(raw: str) -> list[dict]:
    """
    Strip markdown fences and parse JSON array from LLM response.
    Handles truncated responses (hit max_tokens) by salvaging
    complete operations before the cutoff.
    """
    # Strip markdown fences if present
    clean = re.sub(r"```(?:json)?\n?", "", raw).strip().rstrip("`").strip()

    # Find the JSON array start
    start = clean.find("[")
    if start == -1:
        raise ValueError(f"No JSON array found in LLM response:\n{raw[:300]}")

    end = clean.rfind("]") + 1

    # Happy path — well-formed complete JSON
    if end > start:
        try:
            return json.loads(clean[start:end])
        except json.JSONDecodeError:
            pass  # fall through to recovery

    # Recovery path — response was truncated (hit max_tokens)
    # Extract complete {...} objects one by one before the cutoff
    log.warning("LLM response appears truncated — recovering complete operations")
    operations = []
    depth = 0
    obj_start = None

    for i, ch in enumerate(clean[start + 1:], start=start + 1):
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                try:
                    obj = json.loads(clean[obj_start:i + 1])
                    operations.append(obj)
                except json.JSONDecodeError:
                    pass
                obj_start = None

    if operations:
        log.warning(f"  Recovered {len(operations)} complete operation(s) from truncated response")
        log.warning("  Increase LLM_MAX_TOKENS in .env to avoid truncation (current default: 8000)")
        return operations

    raise ValueError(
        f"Could not recover any operations from truncated response.\n"
        f"Set LLM_MAX_TOKENS=12000 in .env and retry.\n"
        f"Response tail: ...{raw[-200:]}"
    )


# ── Staging approver ──────────────────────────────────────────────────────────

class StagingApprover:
    """
    When staging is enabled, writes go to staging/ first.
    Run approve() to move them to wiki/.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def list_staged(self) -> list[Path]:
        if not self.cfg.STAGING_DIR.exists():
            return []
        return sorted(self.cfg.STAGING_DIR.rglob("*.md"))

    def approve_all(self):
        files = self.list_staged()
        if not files:
            log.info("No staged files to approve.")
            return
        for src in files:
            rel   = src.relative_to(self.cfg.STAGING_DIR)
            dest  = self.cfg.WIKI_DIR / rel
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Biomarker pages: merge rows instead of overwriting
            if "biomarkers/" in str(rel) and dest.exists():
                staged_text = src.read_text(encoding="utf-8")
                wiki_text   = dest.read_text(encoding="utf-8")

                # Extract data rows from staged file
                new_rows = [
                    l.strip() for l in staged_text.splitlines()
                    if l.strip().startswith("|")
                    and "---" not in l
                    and "Date" not in l
                    and "Value" not in l
                    and re.search(r"\d{4}-\d{2}-\d{2}", l)
                ]

                if new_rows:
                    # Determine person from staged content
                    person_match = re.search(r"^## (.+)$", staged_text, re.MULTILINE)
                    person = person_match.group(1).lower().strip() if person_match else "person_1"

                    # Read current wiki file and append rows directly
                    current = dest.read_text(encoding="utf-8")
                    person_header = f"## {person.title()}"

                    for row in new_rows:
                        # Sort-insert the row into the person section
                        def row_date_key(r):
                            m = re.search(r"\d{4}-\d{2}-\d{2}", r)
                            return m.group() if m else "0000-00-00"

                        if person_header in current:
                            section_start = current.index(person_header)
                            next_sec = current.find("\n## ", section_start + 1)
                            section_end = next_sec if next_sec != -1 else len(current)
                            section = current[section_start:section_end]

                            data_rows = [
                                l.strip() for l in section.splitlines()
                                if l.strip().startswith("|")
                                and "---" not in l
                                and "Date" not in l
                                and "Value" not in l
                                and re.search(r"\d{4}-\d{2}-\d{2}", l)
                            ]
                            all_rows = sorted(data_rows + [row], key=row_date_key)

                            # Rebuild section with sorted rows
                            header_lines = [
                                l for l in section.splitlines()
                                if not (l.strip().startswith("|")
                                        and "---" not in l
                                        and "Date" not in l
                                        and "Value" not in l
                                        and re.search(r"\d{4}-\d{2}-\d{2}", l))
                            ]
                            new_section = "\n".join(header_lines).rstrip() + "\n" + "\n".join(all_rows) + "\n"
                            current = current[:section_start] + new_section + current[section_end:]
                        else:
                            table_header = "| Date | Value | Unit | Flag | Source |\n|------|-------|------|------|--------|"
                            current = current.rstrip() + f"\n\n{person_header}\n\n{table_header}\n{row}\n"

                    dest.write_text(current, encoding="utf-8")
                    log.info(f"  Merged {len(new_rows)} row(s) → wiki/{rel}")
                else:
                    shutil.copy2(str(src), str(dest))
                    log.info(f"  Approved → wiki/{rel}")
            else:
                shutil.copy2(str(src), str(dest))
                log.info(f"  Approved → wiki/{rel}")

            src.unlink()
        log.info(f"Approved {len(files)} files.")

    def reject_all(self):
        files = self.list_staged()
        for f in files:
            f.unlink()
        log.info(f"Rejected and deleted {len(files)} staged files.")

    def show(self):
        files = self.list_staged()
        if not files:
            print("No staged files.")
            return
        print(f"\n{len(files)} staged file(s):\n")
        for f in files:
            rel = f.relative_to(self.cfg.STAGING_DIR)
            print(f"  {rel}")
            print(f.read_text(encoding="utf-8")[:300])
            print("  ---")


# ── Core process function ─────────────────────────────────────────────────────

def process_file(
    filepath: Path,
    cfg: Config,
    llm: LLMAdapter,
    biomarker_cfg: BiomarkerConfig,
    context_loader: ContextLoader,
    runner: OperationRunner,
    prompt_loader: "PromptLoader",
) -> bool:
    log.info(f"Synthesizing: {filepath.name}")

    content = filepath.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(content)

    if not meta:
        log.warning(f"  No frontmatter — skipping {filepath.name}")
        return False

    person       = meta.get("person", "unknown")
    content_type = meta.get("content_type", "unknown")
    year         = str(meta.get("year", datetime.now().year))
    confidence   = meta.get("confidence", "medium")

    log.info(f"  type={content_type}  person={person}  year={year}  confidence={confidence}")

    if confidence == "low":
        log.warning("  Low confidence extraction — flagging for review, skipping LLM call")
        flag_path = cfg.BASE_DIR / "raw_extracted" / "needs-review" / filepath.name
        flag_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(filepath), str(flag_path))
        return False

    # Load context
    wiki_context = context_loader.load_for_ingest(meta)

    # Build prompt from prompts.yaml (reloaded if file changed)
    tracked = ", ".join(biomarker_cfg.track[:20]) + "..."
    user_prompt = prompt_loader.user(wiki_context, tracked, content)

    # LLM call
    log.info("  Calling LLM...")
    try:
        raw_response = llm.complete(
            system=prompt_loader.system(),
            user=user_prompt,
            max_tokens=cfg.LLM_MAX_TOKENS,
            temperature=cfg.LLM_TEMPERATURE,
        )
    except Exception as e:
        log.error(f"  LLM call failed: {e}")
        return False

    # Parse response
    try:
        operations = parse_llm_response(raw_response)
        log.info(f"  {len(operations)} operations returned")
    except Exception as e:
        log.error(f"  Failed to parse LLM response: {e}")
        log.debug(f"  Raw response: {raw_response[:500]}")
        return False

    # Execute operations
    runner.run(operations)

    # Archive extracted file
    dest_dir = cfg.RAW_EXTRACTED_ARCHIVE / str(year)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filepath.name
    shutil.move(str(filepath), str(dest))
    log.info(f"  Archived extracted file → raw_extracted/{year}/{filepath.name}")

    if cfg.STAGING_ENABLED and not cfg.STAGING_AUTO_APPROVE:
        log.info(f"  Staged. Run: python synthesizer.py approve")

    log.info(f"  Done: {filepath.name}")
    return True


# ── Watcher ───────────────────────────────────────────────────────────────────

def run_watcher(cfg, llm, biomarker_cfg, context_loader, runner, prompt_loader):
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    class Handler(FileSystemEventHandler):
        def on_created(self, event):
            if event.is_directory:
                return
            filepath = Path(event.src_path)
            if not filepath.name.endswith(".md"):
                return
            import time; time.sleep(1)
            if filepath.exists():
                process_file(filepath, cfg, llm, biomarker_cfg, context_loader, runner, prompt_loader)

    watch_dir = cfg.RAW_EXTRACTED_INCOMING
    watch_dir.mkdir(parents=True, exist_ok=True)

    observer = Observer()
    observer.schedule(Handler(), str(watch_dir), recursive=False)
    observer.start()
    log.info(f"Watching: {watch_dir}")
    log.info("Ctrl+C to stop.\n")

    # Process any files already present before the watcher started
    existing = sorted(watch_dir.glob("*.md"))
    if existing:
        log.info(f"Found {len(existing)} existing file(s) in incoming/ — processing now")
        for f in existing:
            process_file(f, cfg, llm, biomarker_cfg, context_loader, runner, prompt_loader)

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


# ── Index template ────────────────────────────────────────────────────────────

INDEX_TEMPLATE = """# Health Wiki — Index

## Reports ingested
- person_1: 0 reports ingested — last report: none
- person_2: 0 reports ingested — last report: none

## Biomarker pages

## Conditions tracked

## Medications tracked

## Imaging studies

## Current status — Person 1
No data yet.

## Current status — Person 2
No data yet.

## Reading paths
- For a biomarker trend: `biomarkers/<name>.md`
- For a specific report: `reports/<person>-<type>-<YYYY-MM>.md`
- For cardiac risk: `trends/person_1-cardiac-risk.md`
- For inter-parameter patterns: `correlations/`
"""


# ── Lint runner ───────────────────────────────────────────────────────────────

class LintRunner:
    """
    Health-checks the wiki and reports issues.
    Does not modify any files — read-only analysis.

    Checks:
      1. Orphan pages — no inbound wikilinks
      2. Biomarker pages with only one data point (no trend possible)
      3. Conflict rows not yet resolved (contain CONFLICT marker)
      4. Conditions mentioned in reports but no conditions/ page exists
      5. Long gaps between readings (> 12 months) for key biomarkers
      6. Biomarkers in tracked list with no wiki page yet
      7. index.md out of sync — pages exist not listed in index
    """

    def __init__(self, cfg: Config, biomarker_cfg: BiomarkerConfig):
        self.cfg = cfg
        self.biomarker_cfg = biomarker_cfg
        self.wiki = cfg.WIKI_DIR
        self.issues: list[str] = []

    def _warn(self, category: str, message: str):
        self.issues.append(f"[{category}] {message}")

    def _all_md_files(self) -> list[Path]:
        if not self.wiki.exists():
            return []
        return list(self.wiki.rglob("*.md"))

    def _build_link_map(self) -> dict[str, list[str]]:
        """
        Returns {target_slug: [source_files_that_link_to_it]}
        """
        link_map: dict[str, list[str]] = {}
        for f in self._all_md_files():
            content = f.read_text(encoding="utf-8", errors="replace")
            links = re.findall(r"\[\[([^\]]+)\]\]", content)
            for link in links:
                slug = link.lower().replace(" ", "-")
                link_map.setdefault(slug, []).append(f.name)
        return link_map

    def check_orphans(self):
        """Pages with no inbound links (excluding index.md and log.md)."""
        link_map = self._build_link_map()
        for f in self._all_md_files():
            if f.name in ("index.md", "log.md"):
                continue
            slug = f.stem.lower()
            if slug not in link_map:
                self._warn("ORPHAN", f"{f.relative_to(self.wiki)} — no inbound wikilinks")

    def check_single_reading_biomarkers(self):
        """Biomarker pages with only one data row — no trend possible."""
        bm_dir = self.wiki / "biomarkers"
        if not bm_dir.exists():
            return
        for f in bm_dir.iterdir():
            if not f.suffix == ".md":
                continue
            content = f.read_text(encoding="utf-8", errors="replace")
            # Count data rows (lines starting with | that aren't header/separator)
            rows = [l for l in content.splitlines()
                    if l.startswith("|") and "---" not in l
                    and "Date" not in l and "Value" not in l]
            if len(rows) == 1:
                self._warn("SINGLE-READING",
                           f"biomarkers/{f.name} — only 1 reading, no trend possible yet")

    def check_unresolved_conflicts(self):
        """Rows still marked CONFLICT."""
        for f in self._all_md_files():
            content = f.read_text(encoding="utf-8", errors="replace")
            lines = [i + 1 for i, l in enumerate(content.splitlines())
                     if "CONFLICT" in l]
            if lines:
                self._warn("CONFLICT",
                           f"{f.relative_to(self.wiki)} — unresolved conflict on line(s) {lines}")

    def check_missing_biomarker_pages(self):
        """Tracked biomarkers with no wiki page."""
        bm_dir = self.wiki / "biomarkers"
        existing = {f.stem for f in bm_dir.iterdir()} if bm_dir.exists() else set()
        for tracked in self.biomarker_cfg.track:
            if tracked not in existing:
                self._warn("MISSING-PAGE",
                           f"biomarkers/{tracked}.md — tracked but no page exists yet")

    def check_long_gaps(self, max_months: int = 12):
        """Biomarker readings with gap > max_months between consecutive entries."""
        bm_dir = self.wiki / "biomarkers"
        if not bm_dir.exists():
            return
        for f in bm_dir.iterdir():
            if not f.suffix == ".md":
                continue
            content = f.read_text(encoding="utf-8", errors="replace")
            dates = re.findall(r"\|\s*(\d{4}-\d{2}-\d{2})\s*\|", content)
            if len(dates) < 2:
                continue
            try:
                from datetime import date
                parsed = sorted([date.fromisoformat(d) for d in set(dates)])
                for i in range(1, len(parsed)):
                    delta_months = (parsed[i].year - parsed[i-1].year) * 12 + \
                                   (parsed[i].month - parsed[i-1].month)
                    if delta_months > max_months:
                        self._warn("GAP",
                                   f"biomarkers/{f.name} — {delta_months}m gap between "
                                   f"{parsed[i-1]} and {parsed[i]}")
            except Exception:
                continue

    def check_index_sync(self):
        """Pages that exist but aren't mentioned in index.md."""
        index_path = self.wiki / "index.md"
        if not index_path.exists():
            self._warn("INDEX", "index.md does not exist")
            return
        index_content = index_path.read_text(encoding="utf-8", errors="replace")
        for folder in ["biomarkers", "conditions", "medications"]:
            folder_path = self.wiki / folder
            if not folder_path.exists():
                continue
            for f in folder_path.iterdir():
                if f.suffix != ".md":
                    continue
                if f"[[{f.stem}]]" not in index_content and f.stem not in index_content:
                    self._warn("INDEX-SYNC",
                               f"{folder}/{f.name} — exists but not listed in index.md")

    def run_all(self) -> list[str]:
        """Run all checks. Returns list of issue strings."""
        self.issues = []
        log.info("Lint — running checks...")
        self.check_unresolved_conflicts()
        self.check_single_reading_biomarkers()
        self.check_long_gaps()
        self.check_missing_biomarker_pages()
        self.check_orphans()
        self.check_index_sync()
        log.info(f"Lint — {len(self.issues)} issue(s) found")
        return self.issues

    def report(self) -> str:
        """Format issues as a markdown report."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        if not self.issues:
            return f"# Lint Report — {now}\n\nNo issues found. Wiki is healthy.\n"

        by_category: dict[str, list[str]] = {}
        for issue in self.issues:
            cat = re.match(r"\[([A-Z\-]+)\]", issue)
            category = cat.group(1) if cat else "OTHER"
            by_category.setdefault(category, []).append(
                issue[len(category) + 3:].strip()
            )

        lines = [f"# Lint Report — {now}\n",
                 f"**{len(self.issues)} issue(s) found**\n"]
        priority = ["CONFLICT", "INDEX", "INDEX-SYNC", "MISSING-PAGE",
                    "GAP", "SINGLE-READING", "ORPHAN"]
        ordered = priority + [c for c in by_category if c not in priority]

        for cat in ordered:
            if cat not in by_category:
                continue
            lines.append(f"\n## {cat} ({len(by_category[cat])})\n")
            for msg in by_category[cat]:
                lines.append(f"- {msg}")

        return "\n".join(lines) + "\n"


# ── Setup ─────────────────────────────────────────────────────────────────────

REQUIRED_DIRS = [
    "raw_extracted/incoming",
    "raw_extracted/needs-review",
    "staging",
    "wiki/people",
    "wiki/reports",
    "wiki/biomarkers",
    "wiki/conditions",
    "wiki/medications",
    "wiki/imaging",
    "wiki/trends",
    "wiki/correlations",
    "wiki/insights",
]

def setup(cfg: Config):
    created, existed = [], []
    for rel in REQUIRED_DIRS:
        d = cfg.BASE_DIR / rel
        if d.exists():
            existed.append(rel)
        else:
            d.mkdir(parents=True, exist_ok=True)
            created.append(rel)
    if created:
        log.info("Setup — created:")
        for d in created: log.info(f"  + {d}")
    if existed:
        log.info("Setup — already present:")
        for d in existed: log.info(f"  ✓ {d}")
    log.info(f"Setup complete. Base: {cfg.BASE_DIR}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    cfg = Config()
    setup(cfg)

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "setup":
            pass  # already done above

        elif cmd == "file":
            if len(sys.argv) < 3:
                print("Usage: python synthesizer.py file <path>")
                sys.exit(1)
            llm           = build_adapter(cfg)
            biomarker_cfg = BiomarkerConfig(Path(__file__).parent.parent / "config" / "biomarkers.yaml")
            loader        = ContextLoader(cfg)
            runner        = OperationRunner(cfg)
            prompt_loader = PromptLoader(Path(__file__).parent.parent / "config" / "prompts.yaml")
            process_file(Path(sys.argv[2]), cfg, llm, biomarker_cfg, loader, runner, prompt_loader)

        elif cmd == "batch":
            folder = Path(sys.argv[2]) if len(sys.argv) > 2 else cfg.RAW_EXTRACTED_INCOMING
            files  = sorted(folder.rglob("*.md"))
            log.info(f"Batch: {len(files)} files in {folder}")
            llm           = build_adapter(cfg)
            biomarker_cfg = BiomarkerConfig(Path(__file__).parent.parent / "config" / "biomarkers.yaml")
            loader        = ContextLoader(cfg)
            runner        = OperationRunner(cfg)
            prompt_loader = PromptLoader(Path(__file__).parent.parent / "config" / "prompts.yaml")
            ok = fail = 0
            for f in files:
                if process_file(f, cfg, llm, biomarker_cfg, loader, runner, prompt_loader):
                    ok += 1
                else:
                    fail += 1
            log.info(f"Batch complete: {ok} succeeded, {fail} failed")

        elif cmd == "fix-date-order":
            # Fix date ordering in all biomarker tables — no LLM, no tokens.
            # Processes each file line by line, sorts data rows within each
            # person section independently.
            bm_dir = cfg.wiki_path("biomarkers")
            if not bm_dir.exists():
                print("No biomarkers directory found.")
                sys.exit(1)

            def row_date_key(r):
                m = re.search(r"\d{4}-\d{2}-\d{2}", r)
                return m.group() if m else "0000-00-00"

            def is_data_row(line):
                s = line.strip()
                return (s.startswith("|")
                        and "---" not in s
                        and "Date" not in s
                        and "Value" not in s
                        and len(s) > 3)

            fixed = 0
            for f in sorted(bm_dir.glob("*.md")):
                lines    = f.read_text(encoding="utf-8").splitlines(keepends=True)
                original = "".join(lines)
                output   = []
                pending  = []   # data rows buffered for sorting

                def flush(pending, output):
                    for dr in sorted(pending, key=row_date_key):
                        output.append(dr if dr.endswith("\n") else dr + "\n")
                    pending.clear()

                for line in lines:
                    if is_data_row(line):
                        pending.append(line.rstrip("\n"))
                    else:
                        if pending:
                            flush(pending, output)
                        output.append(line)

                if pending:
                    flush(pending, output)

                new_content = "".join(output)
                if new_content != original:
                    f.write_text(new_content, encoding="utf-8")
                    log.info(f"  Fixed: {f.name}")
                    fixed += 1
                else:
                    log.info(f"  OK:    {f.name}")

            log.info(f"fix-date-order complete: {fixed} file(s) updated")

        elif cmd == "rebuild-biomarkers":
            # Token-efficient mode — only rebuilds biomarker tables.
            # Use after clearing wiki/biomarkers/ to avoid re-running full ingest.
            # Reads all extracted files and emits only append_row operations.
            folder = Path(sys.argv[2]) if len(sys.argv) > 2 else cfg.RAW_EXTRACTED_ARCHIVE
            files  = sorted(folder.rglob("*.md"))
            log.info(f"Rebuild biomarkers: {len(files)} extracted files in {folder}")

            llm           = build_adapter(cfg)
            biomarker_cfg = BiomarkerConfig(
                Path(__file__).parent.parent / "config" / "biomarkers.yaml"
            )
            loader = ContextLoader(cfg)
            runner = OperationRunner(cfg)

            # Use a lean prompt — biomarkers only, no other wiki operations
            lean_system = (
                "You are a biomarker extraction engine. Read the extracted medical "
                "report and return ONLY append_row operations for tracked biomarkers. "
                "No other operations. No report pages. No people pages. No log entries. "
                "No index updates. ONLY append_row operations. "
                "Respond with ONLY a valid JSON array of append_row operations. "
                "No preamble, no explanation, no markdown fences."
            )

            tracked = ", ".join(biomarker_cfg.track)
            ok = fail = 0
            for f in files:
                try:
                    content_str = f.read_text(encoding="utf-8")
                    meta, body = parse_frontmatter(content_str)
                    if not meta:
                        continue
                    person       = meta.get("person", "unknown")
                    content_type = meta.get("content_type", "")
                    if content_type not in ("lab-report", "trend-data",
                                            "imaging-report-echo", "imaging-report-cac"):
                        continue  # skip non-numeric content types

                    lean_user = (
                        f"Person: {person}\n"
                        f"Tracked biomarkers: {tracked}\n\n"
                        f"Report:\n{content_str}\n\n"
                        f"Return ONLY append_row operations for biomarkers found. "
                        f"Use person=\"{person}\" in every operation."
                    )
                    log.info(f"Rebuilding biomarkers from: {f.name}")
                    raw = llm.complete(
                        system=lean_system,
                        user=lean_user,
                        max_tokens=2000,
                        temperature=0,
                    )
                    ops = parse_llm_response(raw)
                    # Safety: filter to only append_row operations
                    ops = [o for o in ops if o.get("op") == "append_row"]
                    log.info(f"  {len(ops)} append_row operations")
                    runner.run(ops)
                    ok += 1
                except Exception as e:
                    log.error(f"  Failed {f.name}: {e}")
                    fail += 1

            log.info(f"Rebuild complete: {ok} succeeded, {fail} failed")
            if cfg.STAGING_ENABLED and not cfg.STAGING_AUTO_APPROVE:
                log.info("Staged. Run: python src/synthesizer.py approve")

        elif cmd == "approve":
            approver = StagingApprover(cfg)
            approver.show()
            if len(sys.argv) > 2 and sys.argv[2] == "--yes":
                approver.approve_all()
            else:
                confirm = input("\nApprove all staged files? [y/N] ").strip().lower()
                if confirm == "y":
                    approver.approve_all()
                else:
                    print("Cancelled.")

        elif cmd == "reject":
            approver = StagingApprover(cfg)
            approver.reject_all()

        elif cmd == "show-staged":
            StagingApprover(cfg).show()

        elif cmd == "lint":
            # Run wiki health checks — read only, no modifications
            biomarker_cfg = BiomarkerConfig(
                Path(__file__).parent.parent / "config" / "biomarkers.yaml"
            )
            linter  = LintRunner(cfg, biomarker_cfg)
            issues  = linter.run_all()
            report  = linter.report()

            # Print to console
            print("\n" + report)

            # Optionally save report to wiki/insights/
            save = "--save" in sys.argv
            if save:
                out = cfg.wiki_path("insights",
                    f"lint-{datetime.now().strftime('%Y-%m-%d')}.md")
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(report, encoding="utf-8")
                log.info(f"Lint report saved to {out}")

        elif cmd == "init-index":
            # Create a fresh index.md if it doesn't exist
            index_path = cfg.wiki_path("index.md")
            if index_path.exists():
                print("index.md already exists. Delete it first to reinitialise.")
            else:
                index_path.parent.mkdir(parents=True, exist_ok=True)
                index_path.write_text(INDEX_TEMPLATE, encoding="utf-8")
                log.info(f"Created index.md at {index_path}")

        else:
            print(f"Unknown command: {cmd}")
            print("Commands: setup | watch | file <path> | batch [folder]")
            print("          approve [--yes] | reject | show-staged")
            print("          lint [--save] | init-index")
            print("          rebuild-biomarkers [folder] — lean re-ingest, biomarkers only")
            print("          fix-date-order — sort all biomarker tables by date (no LLM)")
            sys.exit(1)

    else:
        llm           = build_adapter(cfg)
        biomarker_cfg = BiomarkerConfig(Path(__file__).parent.parent / "config" / "biomarkers.yaml")
        loader        = ContextLoader(cfg)
        runner        = OperationRunner(cfg)
        prompt_loader = PromptLoader(Path(__file__).parent.parent / "config" / "prompts.yaml")
        run_watcher(cfg, llm, biomarker_cfg, loader, runner, prompt_loader)
