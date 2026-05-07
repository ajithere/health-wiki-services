"""
Unit tests for Service 2 — Synthesizer
Run from service root: python -m pytest tests/
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from synthesizer import (
    BiomarkerConfig,
    parse_frontmatter,
    parse_llm_response,
    OperationRunner,
    Config,
    LintRunner,
    INDEX_TEMPLATE,
)


# ── BiomarkerConfig tests ─────────────────────────────────────────────────────

BIOMARKERS_YAML = Path(__file__).parent.parent / "config" / "biomarkers.yaml"


def test_normalise_ldl_variants():
    cfg = BiomarkerConfig(BIOMARKERS_YAML)
    for variant in ["LDL", "LDL-C", "LDL Cholesterol", "Low Density Lipoprotein"]:
        assert cfg.normalise(variant) == "ldl-cholesterol", f"Failed for: {variant}"

def test_normalise_indian_lab_names():
    cfg = BiomarkerConfig(BIOMARKERS_YAML)
    assert cfg.normalise("SGPT") == "alt"
    assert cfg.normalise("SGOT") == "ast"
    assert cfg.normalise("Haemoglobin") == "hemoglobin"
    assert cfg.normalise("TSH (Ultrasensitive)") == "tsh"

def test_normalise_vitamin_d():
    cfg = BiomarkerConfig(BIOMARKERS_YAML)
    assert cfg.normalise("25-OH Vitamin D") == "vitamin-d"
    assert cfg.normalise("25(OH)D") == "vitamin-d"

def test_normalise_hba1c_variants():
    cfg = BiomarkerConfig(BIOMARKERS_YAML)
    for variant in ["HbA1c", "Glycated Haemoglobin", "A1C"]:
        assert cfg.normalise(variant) == "hba1c", f"Failed for: {variant}"

def test_normalise_unknown_returns_none():
    cfg = BiomarkerConfig(BIOMARKERS_YAML)
    assert cfg.normalise("Random Unknown Test XYZ") is None

def test_normalise_case_insensitive():
    cfg = BiomarkerConfig(BIOMARKERS_YAML)
    assert cfg.normalise("ldl cholesterol") == "ldl-cholesterol"
    assert cfg.normalise("LDL CHOLESTEROL") == "ldl-cholesterol"


# ── Frontmatter parser tests ──────────────────────────────────────────────────

def test_parse_frontmatter_complete():
    content = "---\nperson: person_1\ncontent_type: lab-report\nyear: 2024\n---\n# Body"
    meta, body = parse_frontmatter(content)
    assert meta["person"] == "person_1"
    assert meta["content_type"] == "lab-report"
    assert meta["year"] == 2024
    assert "# Body" in body

def test_parse_frontmatter_missing():
    content = "# No frontmatter here\nJust body text."
    meta, body = parse_frontmatter(content)
    assert meta == {}
    assert "No frontmatter" in body

def test_parse_frontmatter_empty_body():
    content = "---\nperson: person_2\n---\n"
    meta, body = parse_frontmatter(content)
    assert meta["person"] == "person_2"
    assert body == ""


# ── LLM response parser tests ─────────────────────────────────────────────────

def test_parse_clean_json():
    raw = '[{"op": "create", "path": "wiki/reports/x.md", "content": "# X"}]'
    ops = parse_llm_response(raw)
    assert len(ops) == 1
    assert ops[0]["op"] == "create"

def test_parse_json_with_markdown_fences():
    raw = '```json\n[{"op": "append_log", "content": "## log entry"}]\n```'
    ops = parse_llm_response(raw)
    assert ops[0]["op"] == "append_log"

def test_parse_json_with_preamble():
    raw = 'Here are the operations:\n[{"op": "create", "path": "wiki/x.md", "content": "y"}]'
    ops = parse_llm_response(raw)
    assert len(ops) == 1

def test_parse_multiple_operations():
    raw = """[
        {"op": "create", "path": "wiki/reports/r.md", "content": "# R"},
        {"op": "append_row", "path": "wiki/biomarkers/ldl-cholesterol.md",
         "person": "person_1", "row": "| 2024-03 | 112 | mg/dL | ⚠️ | [[r]] |"},
        {"op": "append_log", "content": "## [2024-03] ingest | lab | person_1 | r.pdf"}
    ]"""
    ops = parse_llm_response(raw)
    assert len(ops) == 3
    assert [o["op"] for o in ops] == ["create", "append_row", "append_log"]


# ── OperationRunner tests ─────────────────────────────────────────────────────

def make_runner():
    tmp = Path(tempfile.mkdtemp())
    os.environ["BASE_DIR"] = str(tmp)
    os.environ["STAGING_ENABLED"] = "false"
    cfg = Config()
    cfg.BASE_DIR = tmp
    cfg.WIKI_DIR = tmp / "wiki"
    cfg.STAGING_DIR = tmp / "staging"
    cfg.STAGING_ENABLED = False
    runner = OperationRunner(cfg)
    return runner, tmp


def test_create_new_page():
    runner, tmp = make_runner()
    try:
        ops = [{"op": "create", "path": "wiki/reports/test.md", "content": "# Test"}]
        runner.run(ops)
        assert (tmp / "wiki" / "reports" / "test.md").exists()
    finally:
        shutil.rmtree(tmp)


def test_create_biomarker_page_on_first_append():
    runner, tmp = make_runner()
    try:
        ops = [{
            "op": "append_row",
            "path": "wiki/biomarkers/ldl-cholesterol.md",
            "person": "person_1",
            "row": "| 2024-03-15 | 112 | mg/dL | ⚠️ | [[person_1-lipid-2024-03]] |"
        }]
        runner.run(ops)
        page = tmp / "wiki" / "biomarkers" / "ldl-cholesterol.md"
        assert page.exists()
        content = page.read_text()
        assert "## Person_1" in content
        assert "| 2024-03-15 | 112 |" in content
    finally:
        shutil.rmtree(tmp)


def test_append_row_second_person():
    runner, tmp = make_runner()
    try:
        ops = [
            {"op": "append_row", "path": "wiki/biomarkers/ldl-cholesterol.md",
             "person": "person_1", "row": "| 2024-03-15 | 112 | mg/dL | ⚠️ | [[r1]] |"},
            {"op": "append_row", "path": "wiki/biomarkers/ldl-cholesterol.md",
             "person": "person_2", "row": "| 2024-01-10 | 118 | mg/dL | ⚠️ | [[r2]] |"},
        ]
        runner.run(ops)
        content = (tmp / "wiki" / "biomarkers" / "ldl-cholesterol.md").read_text()
        assert "## Person_1" in content
        assert "## Person_2" in content
    finally:
        shutil.rmtree(tmp)


def test_conflict_detection_same_date():
    runner, tmp = make_runner()
    try:
        ops = [
            {"op": "append_row", "path": "wiki/biomarkers/ldl-cholesterol.md",
             "person": "person_1", "row": "| 2024-03-15 | 112 | mg/dL | ⚠️ | [[r1]] |"},
            {"op": "append_row", "path": "wiki/biomarkers/ldl-cholesterol.md",
             "person": "person_1", "row": "| 2024-03-15 | 115 | mg/dL | ⚠️ | [[r2]] |"},
        ]
        runner.run(ops)
        content = (tmp / "wiki" / "biomarkers" / "ldl-cholesterol.md").read_text()
        assert "CONFLICT" in content
    finally:
        shutil.rmtree(tmp)


def test_append_log():
    runner, tmp = make_runner()
    try:
        ops = [{"op": "append_log",
                "content": "## [2024-03-15] ingest | lab-report | person_1 | r.pdf"}]
        runner.run(ops)
        log = (tmp / "wiki" / "log.md").read_text()
        assert "lab-report" in log
    finally:
        shutil.rmtree(tmp)


def test_update_section():
    runner, tmp = make_runner()
    try:
        # Create a page first
        page = tmp / "wiki" / "people" / "person_1.md"
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text("# Person_1\n\n## Last updated\n2024-01-01\n\n## Conditions\nNone\n")

        ops = [{"op": "update", "path": "wiki/people/person_1.md",
                "section": "Last updated", "content": "2026-05-06"}]
        runner.run(ops)
        content = page.read_text()
        assert "2026-05-06" in content
        assert "2024-01-01" not in content
    finally:
        shutil.rmtree(tmp)


# ── Index tests ───────────────────────────────────────────────────────────────

def test_init_index_creates_file():
    runner, tmp = make_runner()
    try:
        ops = [{"op": "update_index",
                "person": "person_1",
                "report_type": "lab-report",
                "date": "2024-03-15",
                "new_pages": ["wiki/reports/person_1-lipid-2024-03.md"],
                "new_biomarkers": ["ldl-cholesterol"],
                "status_update": "LDL: 112 mg/dL ⚠️"}]
        runner.run(ops)
        index = (tmp / "wiki" / "index.md")
        assert index.exists()
        content = index.read_text()
        assert "2024-03-15" in content
        assert "LDL: 112" in content
    finally:
        shutil.rmtree(tmp)


def test_update_index_adds_biomarker_link():
    runner, tmp = make_runner()
    try:
        # First ingest — creates index
        runner.run([{"op": "update_index",
                     "person": "person_1", "report_type": "lab-report",
                     "date": "2024-03-15", "new_pages": [],
                     "new_biomarkers": ["ldl-cholesterol"],
                     "status_update": "LDL: 112"}])
        # Second ingest — adds new biomarker
        runner.run([{"op": "update_index",
                     "person": "person_1", "report_type": "lab-report",
                     "date": "2024-06-10", "new_pages": [],
                     "new_biomarkers": ["hba1c"],
                     "status_update": "HbA1c: 5.8"}])
        content = (tmp / "wiki" / "index.md").read_text()
        assert "[[ldl-cholesterol]]" in content
        assert "[[hba1c]]" in content
    finally:
        shutil.rmtree(tmp)


def test_update_index_no_duplicate_biomarker():
    runner, tmp = make_runner()
    try:
        for _ in range(3):
            runner.run([{"op": "update_index",
                         "person": "person_1", "report_type": "lab-report",
                         "date": "2024-03-15", "new_pages": [],
                         "new_biomarkers": ["ldl-cholesterol"],
                         "status_update": "LDL: 112"}])
        content = (tmp / "wiki" / "index.md").read_text()
        assert content.count("[[ldl-cholesterol]]") == 1
    finally:
        shutil.rmtree(tmp)


# ── Lint tests ────────────────────────────────────────────────────────────────

def make_lint_setup():
    tmp = Path(tempfile.mkdtemp())
    os.environ["BASE_DIR"] = str(tmp)
    os.environ["STAGING_ENABLED"] = "false"
    cfg = Config()
    cfg.BASE_DIR = tmp
    cfg.WIKI_DIR = tmp / "wiki"
    cfg.STAGING_DIR = tmp / "staging"
    cfg.STAGING_ENABLED = False
    biomarker_cfg = BiomarkerConfig(BIOMARKERS_YAML)
    return cfg, biomarker_cfg, tmp


def test_lint_detects_conflict():
    cfg, bcfg, tmp = make_lint_setup()
    try:
        bm = tmp / "wiki" / "biomarkers" / "ldl-cholesterol.md"
        bm.parent.mkdir(parents=True, exist_ok=True)
        bm.write_text("# LDL\n\n## Person_1\n\n| Date | Value |\n|---|---|\n"
                      "| 2024-03-15 | 112 | ⚠️ CONFLICT — date already exists |\n")
        linter = LintRunner(cfg, bcfg)
        issues = linter.run_all()
        assert any("CONFLICT" in i for i in issues)
    finally:
        shutil.rmtree(tmp)


def test_lint_detects_single_reading():
    cfg, bcfg, tmp = make_lint_setup()
    try:
        bm = tmp / "wiki" / "biomarkers" / "tsh.md"
        bm.parent.mkdir(parents=True, exist_ok=True)
        bm.write_text("# TSH\n\n## Person_2\n\n| Date | Value | Unit | Flag | Source |\n"
                       "|------|-------|------|------|--------|\n"
                       "| 2024-01-10 | 3.2 | mIU/L | | [[person_2-thyroid-2024-01]] |\n")
        linter = LintRunner(cfg, bcfg)
        issues = linter.run_all()
        assert any("SINGLE-READING" in i for i in issues)
    finally:
        shutil.rmtree(tmp)


def test_lint_detects_orphan():
    cfg, bcfg, tmp = make_lint_setup()
    try:
        page = tmp / "wiki" / "conditions" / "person_1-some-condition.md"
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text("# Some Condition\n\nNo links to this page anywhere.\n")
        linter = LintRunner(cfg, bcfg)
        issues = linter.run_all()
        assert any("ORPHAN" in i for i in issues)
    finally:
        shutil.rmtree(tmp)


def test_lint_detects_gap():
    cfg, bcfg, tmp = make_lint_setup()
    try:
        bm = tmp / "wiki" / "biomarkers" / "ldl-cholesterol.md"
        bm.parent.mkdir(parents=True, exist_ok=True)
        bm.write_text("# LDL\n\n## Person_1\n\n| Date | Value | Unit | Flag | Source |\n"
                       "|------|-------|------|------|--------|\n"
                       "| 2021-01-10 | 145 | mg/dL | ⚠️ | [[r1]] |\n"
                       "| 2024-03-15 | 112 | mg/dL | ⚠️ | [[r2]] |\n")
        linter = LintRunner(cfg, bcfg)
        issues = linter.run_all()
        assert any("GAP" in i for i in issues)
    finally:
        shutil.rmtree(tmp)


def test_lint_report_format():
    cfg, bcfg, tmp = make_lint_setup()
    try:
        # Empty wiki — only missing page issues expected
        (tmp / "wiki").mkdir(parents=True, exist_ok=True)
        linter = LintRunner(cfg, bcfg)
        linter.run_all()
        report = linter.report()
        assert "# Lint Report" in report
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    tests = [
        test_normalise_ldl_variants,
        test_normalise_indian_lab_names,
        test_normalise_vitamin_d,
        test_normalise_hba1c_variants,
        test_normalise_unknown_returns_none,
        test_normalise_case_insensitive,
        test_parse_frontmatter_complete,
        test_parse_frontmatter_missing,
        test_parse_frontmatter_empty_body,
        test_parse_clean_json,
        test_parse_json_with_markdown_fences,
        test_parse_json_with_preamble,
        test_parse_multiple_operations,
        test_create_new_page,
        test_create_biomarker_page_on_first_append,
        test_append_row_second_person,
        test_conflict_detection_same_date,
        test_append_log,
        test_update_section,
        test_init_index_creates_file,
        test_update_index_adds_biomarker_link,
        test_update_index_no_duplicate_biomarker,
        test_lint_detects_conflict,
        test_lint_detects_single_reading,
        test_lint_detects_orphan,
        test_lint_detects_gap,
        test_lint_report_format,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
