# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Kaggle Playground Series Season 6 Episode 6 — stellar object classification (GALAXY / QSO / STAR) from photometric measurements. The competition metric is **balanced accuracy**.

## Environment
- Package manager: uv
- Run scripts: `uv run python <python_file>.py`
- Add packages: `uv pip install <package>`
- Never use `pip install` or `python` directly
- Launch Jupyter for the notebook: `uv run jupyter notebook`

## Environment

- Python 3.11, managed via **uv** (`pyproject.toml` + `uv.lock`).
- Install dependencies: `uv sync`
- Run the placeholder script: `uv run python main.py`
- Launch Jupyter for the notebook: `uv run jupyter notebook`

## Data

| File | Description |
|---|---|
| `data/train.csv` | 577 347 rows × 11 cols (features + `class`) |
| `data/test.csv` | 247 435 rows × 10 cols (no `class`) |
| `data/sample_submission.csv` | id + class template |

Raw features: `alpha`, `delta`, photometric bands `u g r i z`, `redshift`, `spectral_type` (categorical), `galaxy_population` (categorical). Target: `class` → mapped to `{GALAXY:0, QSO:1, STAR:2}`.

## Submissions
- `submissions\oof_tabpfn.parquet` - oof probability scores for tabpfn
- `submissions\oof_test_abpfn.parquet` - average test probability scores of five TabPFN models run in CV
- `submissions\oof_xgb.parquet` - oof probability scores for xgb
- `submissions\oof_test_xgb.parquet` - average test probability scores of five XGB models run in CV

## Code Style
- Python only; functions small and single purpose
- Prefer explicit over clever; readability matters for iteration speed
- Do not overengineer - keep code as straightforward as possible.
- No target leakage ever
- Always lowercase column names `df.columns = df.columns.str.lower()`

## General
- Only do what specifically is asked for.
- Do not over-engineer. Keep things as simple as possible.
- Readability over cleverness.
- No target leakage ever.


