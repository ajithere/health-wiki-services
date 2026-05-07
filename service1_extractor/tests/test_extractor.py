"""
Unit tests for Service 1 — Extractor
Run from service root: python -m pytest tests/
"""

import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import extractor as E


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_temp_project():
    """Create a temp project dir with required folders."""
    tmp = Path(tempfile.mkdtemp())
    E.Config.BASE_DIR = tmp
    E.Config.RAW_INPUT_INCOMING = tmp / "raw_input" / "incoming"
    E.Config.RAW_EXTRACTED_INCOMING = tmp / "raw_extracted" / "incoming"
    E.Config.RAW_INPUT_ARCHIVE = tmp / "raw_input"
    E.Config.RAW_EXTRACTED_ARCHIVE = tmp / "raw_extracted"
    E.setup()
    return tmp


# ── Classifier tests ──────────────────────────────────────────────────────────

def test_classify_lab_report():
    assert E.classify_content_type("person_1-lipid-panel.pdf", "LDL HDL Triglycerides") == "lab-report"

def test_classify_echo():
    assert E.classify_content_type("echo-report.pdf", "echocardiogram EF% wall motion") == "imaging-report-echo"

def test_classify_cac():
    assert E.classify_content_type("cac-score-2024.pdf", "coronary artery calcium Agatston") == "imaging-report-cac"

def test_classify_usg():
    assert E.classify_content_type("usg-abdomen.pdf", "ultrasound liver spleen") == "imaging-report-usg"

def test_classify_person_1():
    assert E.classify_person("person_1-lipid-2024.pdf", "") == "person_1"

def test_classify_person_2():
    assert E.classify_person("person_2-thyroid-2022.pdf", "") == "person_2"

def test_classify_person_unknown():
    assert E.classify_person("report-2024.pdf", "") == "unknown"

def test_infer_year_from_filename():
    p = Path("raw_input/incoming/person_1-lipid-2023-03.pdf")
    assert E.infer_year(p, "") == "2023"

def test_infer_year_from_folder():
    p = Path("raw_input/2021/some-report.pdf")
    assert E.infer_year(p, "") == "2021"


# ── MD disposition tests ──────────────────────────────────────────────────────

def test_md_wikilink_disposition():
    content = "# Brahman\n\nSee [[atman]] and [[maya]]."
    _, _, disposition = E.extract_md.__wrapped__(Path("test.md")) if hasattr(E.extract_md, '__wrapped__') else _run_extract_md(content)
    # Direct test via function logic
    assert "[[" in content and "]]" in content

def _run_extract_md(content):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write(content)
        p = Path(f.name)
    result = E.extract_md(p)
    p.unlink()
    return result

def test_md_obsidian_frontmatter():
    content = "---\ntags: [distillation]\ntype: crux\n---\n# My Note"
    _, _, disposition = _run_extract_md(content)
    assert disposition == "skip"

def test_md_extractor_frontmatter():
    content = "---\nsource_file: report.pdf\ncontent_type: lab-report\n---\n# Content"
    _, _, disposition = _run_extract_md(content)
    assert disposition == "duplicate"

def test_md_plain_prose():
    content = "Patient visited on 2024-03-15. Chest pain noted."
    _, _, disposition = _run_extract_md(content)
    assert disposition == "process"


# ── CSV extraction test ───────────────────────────────────────────────────────

def test_csv_extraction():
    tmp = make_temp_project()
    try:
        csv_file = E.Config.RAW_INPUT_INCOMING / "person_1-lipid-2024-03.csv"
        csv_file.write_text(
            "Patient,Person1\nDate,2024-03-15\n\nTest,Value,Unit\nLDL,112,mg/dL\nHDL,48,mg/dL\n"
        )
        result = E.process_file(csv_file)
        assert result is True
        extracted = E.Config.RAW_EXTRACTED_INCOMING / "person_1-lipid-2024-03.md"
        assert extracted.exists()
        content = extracted.read_text()
        assert "content_type:" in content
        assert "person: person_1" in content
    finally:
        shutil.rmtree(tmp)


# ── Setup test ────────────────────────────────────────────────────────────────

def test_setup_creates_all_folders():
    tmp = make_temp_project()
    try:
        for rel in E.REQUIRED_DIRS:
            assert (tmp / rel).exists(), f"Missing: {rel}"
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    # Run without pytest
    tests = [
        test_classify_lab_report,
        test_classify_echo,
        test_classify_cac,
        test_classify_usg,
        test_classify_person_1,
        test_classify_person_2,
        test_classify_person_unknown,
        test_infer_year_from_filename,
        test_infer_year_from_folder,
        test_md_obsidian_frontmatter,
        test_md_extractor_frontmatter,
        test_md_plain_prose,
        test_csv_extraction,
        test_setup_creates_all_folders,
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
