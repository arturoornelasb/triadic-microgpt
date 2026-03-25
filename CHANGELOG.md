# Changelog

All notable changes to this project will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - 2026-03-24

### Added
- **triadic-head** published to PyPI (`pip install triadic-head`)
- OIDC Trusted Publishing via GitHub Actions
- CI workflow (tests on Python 3.10, 3.11, 3.12)
- Full badge set: CI, PyPI, Python, License, HuggingFace, DOI
- Zenodo integration for software archival
- Data download instructions (data/README.md)
- Community files: CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md
- Issue and PR templates
- GitHub Discussions enabled

### Fixed
- Removed internal AI agent config files (.claude/, .agents/, CLAUDE.md) from repository
- Fixed repo structure in README (removed duplicates and gitignored directories)

### Research
- 11 experiments documented (60+ runs)
- Production model: Run 15 (v1.4-strongalign), 40M params
- Paper: 27 pages, all experiments included
- Two HuggingFace models: triadic-gpt-40m, triadic-gpt2-medium

## [0.9-beta] - 2026-03-20

### Added
- D-A19: GPT-2 355M with restored algebra
- D-A18: reptimeline integration
- Beta snapshot before public release refactor
