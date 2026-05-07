"""
Audit script — run this against your actual wiki folder.
Usage: python audit_wiki.py /path/to/my_health_project/wiki
"""
import sys
import re
from pathlib import Path
from collections import defaultdict

wiki = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("wiki")

if not wiki.exists():
    print(f"Wiki not found at {wiki}")
    sys.exit(1)

issues = []
stats  = defaultdict(int)

# ── Biomarker audit ───────────────────────────────────────────────────────────
bm_dir = wiki / "biomarkers"
if bm_dir.exists():
    for f in sorted(bm_dir.iterdir()):
        if f.suffix != ".md":
            continue
        content = f.read_text(encoding="utf-8", errors="replace")
        stats["biomarker_pages"] += 1

        # Count data rows per person
        sections = re.split(r"\n## ", content)
        for section in sections[1:]:  # skip title
            person = section.split("\n")[0].strip().lower()
            rows = [
                l for l in section.splitlines()
                if l.startswith("|")
                and "---" not in l
                and "Date" not in l
                and "Value" not in l
            ]
            stats["biomarker_rows"] += len(rows)
            if len(rows) == 1:
                issues.append(f"SINGLE-ROW  biomarkers/{f.name} [{person}] — only 1 reading (may be truncated)")
            if len(rows) == 0:
                issues.append(f"EMPTY-TABLE biomarkers/{f.name} [{person}] — table exists but no data rows")

            # Extract dates from FIRST CELL ONLY (avoid dates in flag/note columns)
            dates = []
            for r in rows:
                cells = r.split("|")
                if len(cells) > 1:
                    m = re.search(r"\d{4}-\d{2}-\d{2}", cells[1])
                    if m:
                        dates.append(m.group())

            if dates != sorted(dates):
                issues.append(f"DATE-ORDER  biomarkers/{f.name} [{person}] — dates not in order: {dates}")

            # Check for duplicate dates
            if len(dates) != len(set(dates)):
                dupes = [d for d in set(dates) if dates.count(d) > 1]
                issues.append(f"DUPE-DATE   biomarkers/{f.name} [{person}] — duplicate dates: {dupes}")

# ── Reports audit ─────────────────────────────────────────────────────────────
rep_dir = wiki / "reports"
if rep_dir.exists():
    for f in sorted(rep_dir.iterdir()):
        if f.suffix != ".md":
            continue
        stats["report_pages"] += 1
        content = f.read_text(encoding="utf-8", errors="replace")
        # Check has at least one wikilink
        if "[[" not in content:
            issues.append(f"NO-LINKS    reports/{f.name} — no wikilinks (orphan risk)")

# ── People audit ──────────────────────────────────────────────────────────────
ppl_dir = wiki / "people"
if ppl_dir.exists():
    for f in sorted(ppl_dir.iterdir()):
        if f.suffix != ".md":
            continue
        stats["people_pages"] += 1
        content = f.read_text(encoding="utf-8", errors="replace")
        wikilinks = re.findall(r"\[\[([^\]]+)\]\]", content)
        stats["people_wikilinks"] += len(wikilinks)
        if len(wikilinks) < 3:
            issues.append(f"FEW-LINKS   people/{f.name} — only {len(wikilinks)} wikilinks (may be incomplete)")

# ── Log audit ─────────────────────────────────────────────────────────────────
log_file = wiki / "log.md"
if log_file.exists():
    log_content = log_file.read_text(encoding="utf-8", errors="replace")
    ingest_entries = re.findall(r"## \[\d{4}-\d{2}-\d{2}\] ingest", log_content)
    stats["log_entries"] = len(ingest_entries)
else:
    issues.append("MISSING     log.md — not found")

# ── Index audit ───────────────────────────────────────────────────────────────
index_file = wiki / "index.md"
if index_file.exists():
    index_content = index_file.read_text(encoding="utf-8", errors="replace")
    bm_links = re.findall(r"\[\[[a-z][a-z0-9\-]+\]\]", index_content)
    stats["index_biomarker_links"] = len(bm_links)
else:
    issues.append("MISSING     index.md — not found")

# ── Cross-reference check ─────────────────────────────────────────────────────
# Build map of all pages and all links
all_pages = {f.stem for f in wiki.rglob("*.md")}
all_links  = set()
for f in wiki.rglob("*.md"):
    content = f.read_text(encoding="utf-8", errors="replace")
    for link in re.findall(r"\[\[([^\]]+)\]\]", content):
        all_links.add(link.lower().replace(" ", "-"))

broken_links = all_links - {p.lower() for p in all_pages}
for bl in sorted(broken_links)[:10]:  # show first 10 only
    issues.append(f"BROKEN-LINK [[{bl}]] — linked but no page exists")
if len(broken_links) > 10:
    issues.append(f"BROKEN-LINK ... and {len(broken_links)-10} more broken links")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== WIKI AUDIT REPORT ===\n")
print("Stats:")
for k, v in sorted(stats.items()):
    print(f"  {k:<30} {v}")

print(f"\n{'Issues found: ' + str(len(issues)) if issues else 'No issues found'}")
if issues:
    # Group by type
    by_type = defaultdict(list)
    for issue in issues:
        itype = issue.split()[0]
        by_type[itype].append(issue)

    priority = ["EMPTY-TABLE", "DUPE-DATE", "DATE-ORDER", "BROKEN-LINK",
                "MISSING", "NO-LINKS", "FEW-LINKS", "SINGLE-ROW"]
    ordered = priority + [t for t in by_type if t not in priority]

    for itype in ordered:
        if itype not in by_type:
            continue
        print(f"\n  [{itype}] ({len(by_type[itype])})")
        for msg in by_type[itype]:
            print(f"    {msg}")

print("\n=== END AUDIT ===")
