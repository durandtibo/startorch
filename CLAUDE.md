# CLAUDE.md

Guidelines for coding agents working in this repository.

## Project

Shared, reusable Makefiles for formatting and linting common file types (YAML, Markdown, shell,
TOML, Makefiles, GitHub Actions). Consumers vendor this repo as a git subtree (conventionally at
`.make/`) and `include` the `.mk` files they need.

- `*.mk` — one self-contained module per file type or concern (`yaml.mk`, `shell.mk`, `toml.mk`,
  `markdown.mk`, `makefile.mk`, `actions.mk`, `uv.mk`), plus helpers (`prettier.mk`, `help.mk`,
  `self.mk`).
- `Makefile` — the repo's own entry point; includes every module and defines `install-tools`,
  `format` and `lint`.
- `testdata/` — sample files used by CI to exercise each module.
- `tests/` — shell tests (e.g. `tests/test-self-mk.sh` for `update-subtree`).
- `.github/workflows/` — CI (`ci-test.yml` runs each module against `testdata/`).

## Environment

- Tools (`prettier`, `yamllint`, `mbake`, `checkmake`, `shellcheck`, `shfmt`, `markdownlint`,
  `actionlint`, ...) are installed on demand by `install-*` targets; run `make install-tools` to
  install them all.
- Run `make help` to list documented targets.
- Before committing, run `make format` and `make lint`.
- Run `bash tests/test-self-mk.sh` when touching `self.mk`.

## Workflow

- One branch and one PR per logical change, branched from an up-to-date `main`.
- Name branches `<type>/<short-description>` in kebab-case (feat, bugfix, docs, refactor, test,
  chore, ci).
- Never commit or push directly to `main`.
- Resolve conflicts by merging `origin/main` into the branch, not by rebasing and force-pushing.
- PR descriptions state what changed, what was deliberately left alone, and any behaviour change
  (renamed targets, changed variable defaults) since consumers depend on them.
- Follow `.github/PULL_REQUEST_TEMPLATE.md` and `CONTRIBUTING.md`.

## Makefile conventions

- Every `.mk` file is include-guarded:
  `ifndef FOO_MK_INCLUDED` / `FOO_MK_INCLUDED := 1` ... `endif`.
- Document optional variables in a header comment and define them with `?=` so consumers can
  override them before including.
- Include sibling modules via `$(dir $(lastword $(MAKEFILE_LIST)))other.mk` so the file works from
  any vendored path.
- Mark every target `.PHONY`.
- Public targets carry a `## description` comment so they appear in `make help`. Internal
  `install-*` targets do not.
- Each `format-*`/`lint-*` target depends on an `install-*` target that installs the tool only if
  it is missing from `PATH` (Homebrew on Darwin, a sensible fallback elsewhere).
- Keep modules independent: a consumer must be able to include only the files they need.
- Match the existing output style (emoji-prefixed `@echo` start/finish messages).
- Recipes must work on both macOS and Linux; avoid GNU-only flags.
- Never hardcode paths to the consumer's layout; expose them as variables.
- When adding or renaming a module or target, update `README.md` and the root `Makefile`
  (`install-tools`, `format`, `lint`).

## Tests and CI

- Each module has a CI job in `.github/workflows/ci-test.yml` that runs its targets against files
  in `testdata/`. Add a sample file and a job for any new module.
- Shell tests live in `tests/`; they use temp dirs and must not touch the real repo or network.
- Pin GitHub Actions by commit SHA (see `bot-pin-action.yaml`) and keep workflow `permissions`
  minimal and explicit.

## Reporting

- State which commands and tests were run and which were not, and that CI was not awaited.
- Report failures that already exist on `main` rather than hiding them.
