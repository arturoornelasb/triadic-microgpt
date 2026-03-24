# Contributing to TriadicGPT

Thanks for your interest in contributing! This project uses the [BUSL-1.1](LICENSE) license. By contributing, you agree to the terms in [TERMS.md](TERMS.md).

## Getting Started

1. Fork the repository
2. Clone your fork and set up the environment:
   ```bash
   git clone https://github.com/<your-username>/triadic-microgpt.git
   cd triadic-microgpt
   conda env create -f environment.yml
   conda activate triadic-microgpt
   git lfs install && git lfs pull
   ```
3. Create a branch: `git checkout -b my-feature`

## Development

Run tests and linting before submitting:

```bash
python tests/test_all.py          # 37 unit tests
cd triadic-head && pytest tests/  # 33 triadic-head tests
ruff check .                      # Linting
```

## Pull Requests

- Keep PRs focused on a single change
- Include tests for new functionality
- Update documentation if behavior changes
- Reference related issues with `Fixes #N` or `Closes #N`

## Reporting Bugs

Use the [bug report template](https://github.com/arturoornelasb/triadic-microgpt/issues/new?template=bug_report.yml). Include a minimal reproduction case.

## Questions

For general questions, use [GitHub Discussions](https://github.com/arturoornelasb/triadic-microgpt/discussions).

## Commercial Use

This project is BUSL-1.1 licensed. Individuals, academics, and non-profits can use it freely. For commercial use, see [COMMERCIAL.md](COMMERCIAL.md) or contact arturoornelas62@gmail.com.
