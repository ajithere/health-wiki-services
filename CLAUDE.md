# Health Wiki Services

Two microservices to build and maintain a personal health wiki
from medical documents.

## Project structure

service1_extractor/   — File extractor (no LLM)
service2_synthesizer/ — Wiki synthesizer (LLM-powered)
raw_input/            — Source documents (read only)
raw_extracted/        — Extractor output
wiki/                 — LLM-maintained wiki
staging/              — Staged wiki writes pending approval

## Key files

- service1_extractor/src/extractor.py
- service2_synthesizer/src/synthesizer.py
- service2_synthesizer/config/prompts.yaml   ← tune LLM behaviour here
- service2_synthesizer/config/biomarkers.yaml ← tracked parameters

## Running

Terminal 1: cd service1_extractor && python3 src/extractor.py
Terminal 2: cd service2_synthesizer && python3 src/synthesizer.py

## Current status

Core pipeline working. Next: build correlation sweep feature.
See service2_synthesizer/src/synthesizer.py for the synthesizer.
See service2_synthesizer/config/prompts.yaml for all LLM prompts.
