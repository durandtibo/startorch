SHELL=/bin/bash
NAME=startorch
SOURCE=src/$(NAME)
TESTS=tests
UNIT_TESTS=tests/unit
INTEGRATION_TESTS=tests/integration

.PHONY : conda
conda :
	conda env create -f environment.yaml --force

.PHONY : config-poetry
config-poetry :
	poetry config experimental.system-git-client true
	poetry config --list

.PHONY : install
install :
	poetry install --no-interaction --all-extras

.PHONY : install-all
install-all :
	poetry install --no-interaction --all-extras --with exp

.PHONY : install-min
install-min :
	poetry install --no-interaction

.PHONY : update
update :
	-poetry self update
	poetry update
	-pre-commit autoupdate

.PHONY : lint
lint :
	ruff check --output-format=github .

.PHONY : format
format :
	black --check .

.PHONY : docformat
docformat :
	docformatter --config ./pyproject.toml --in-place $(SOURCE)

.PHONY : doctest-src
doctest-src :
	python -m pytest --xdoctest $(SOURCE)

.PHONY : test
test :
	python -m pytest

.PHONY : unit-test
unit-test :
	python -m pytest --xdoctest --timeout 10 $(UNIT_TESTS)

.PHONY : unit-test-cov
unit-test-cov :
	python -m pytest --xdoctest --timeout 10 --cov-report html --cov-report xml --cov-report term --cov=$(NAME) $(UNIT_TESTS)

.PHONY : publish-pypi
publish-pypi :
	poetry config pypi-token.pypi ${PYPI_TOKEN}
	poetry publish --build
