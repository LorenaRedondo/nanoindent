# Contributing to nanoindent

Thank you for your interest. nanoindent is alpha software maintained on a
best-effort basis; issues and pull requests are welcome.

## Ground rules

- Be kind. This project follows the [Code of Conduct](CODE_OF_CONDUCT.md).
- **Never commit measured or instrument data.** The repository holds code and
  synthetic examples only. Tests and examples must generate their own data. The
  `.gitignore` blocks common raw-data formats; do not work around it.
- Do not contribute code you cannot license under the EUPL-1.2. Copied or
  adapted third-party code needs a compatible licence, and its notice must be
  kept. If in doubt, open an issue first.
- By contributing you agree your work is released under the
  [EUPL-1.2](LICENSE).

## Development setup

You need [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
git clone https://github.com/LorenaRedondo/nanoindent.git
cd nanoindent
uv sync                              # installs the package and the dev group from uv.lock
uv run pytest -m "not slow"          # fast suite, ~20 s
uv run pytest -m slow                # physics validation
```

Dependencies live in `pyproject.toml`. Runtime dependencies go under
`[project] dependencies`, optional features under `[project.optional-dependencies]`
and development tools under `[dependency-groups] dev`. After changing them run
`uv lock` and commit `uv.lock`.

## Making a change

1. Open an issue for anything beyond a small fix, so the approach can be agreed.
2. Branch from `main`, make focused commits and add or update tests.
3. Run the fast suite with coverage. CI enforces an 85% floor:
   `uv run pytest -m "not slow" --cov=nanoindent --cov-fail-under=85`.
4. Open a pull request using the template and describe what changed and why.

## Style

- Physics claims need a test against an analytic result or a documented
  reference. Say what a number is and where it comes from.
- No magic numbers in solver code: parametrise and document judgement calls.
- Keep units explicit (nm, MPa, pN unless stated).

## Reporting bugs

Include the nanoindent version, Python version, a minimal script that
reproduces the problem and the full traceback. Do not attach real measurement
files; a synthetic reproduction is enough.
