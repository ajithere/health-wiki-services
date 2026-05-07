# Health Wiki — CLAUDE.md

## What this is

A personal health wiki built from medical documents. The extractor
(Service 1) converts raw files into structured markdown. The synthesizer
(Service 2) uses an LLM to create and maintain wiki pages from them.

---

## Directory structure

```
/
├── raw_input/              # Original files — NEVER modify
│   ├── incoming/           # Drop zone
│   └── <year>/             # Archived after extraction
│
├── raw_extracted/          # Service 1 output — structured text
│   ├── incoming/           # Service 2 watches this
│   └── <year>/             # Archived after synthesis
│
└── wiki/
    ├── index.md
    ├── log.md
    ├── people/
    ├── reports/
    ├── biomarkers/
    ├── conditions/
    ├── medications/
    ├── imaging/
    ├── trends/
    ├── correlations/
    └── insights/
```

---

## Persons

Edit this section to match your own tracked persons.
Example structure:

**Person 1** (e.g. primary person)
- Key health focus areas
- Current medications if relevant
- Any constraints on recommendations

**Person 2** (e.g. family member)
- Key health focus areas
- Current medications if relevant

Add person detection patterns to `config/content_type_rules.yaml`
under `person_rules`.

---

## Conventions

- Filenames: `<person>-<type>-<YYYY-MM>.md`
- Wikilinks: `[[filename-without-extension]]`
- Dates: ISO format — `2024-03-15`
- Flag out-of-range values: ⚠️
- Flag significant change from prior: →↑ or →↓
- Log entry format:
  `## [YYYY-MM-DD] ingest | <type> | <person> | <source file>`

---

## Context navigation

1. Always read `wiki/index.md` first
2. Never read `raw_input/` — use `raw_extracted/` or wiki pages
3. Do not answer from general knowledge when a wiki page exists

---

## Conflict resolution

- Same source file re-ingested → update in place
- New file, new date → append to biomarker tables
- Two files, same date, different values → write both, flag ⚠️ conflict
- Never silently overwrite — log every change

---

## Clinical tone

- Evidence-based and precise
- Never diagnose — report findings and flag only
- Flag out-of-range values with ⚠️
- Note both mg/dL and mmol/L when source uses either
- When uncertain about a reference range, note the uncertainty
