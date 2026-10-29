FUNCTIONS	?= data/input/functions_definition.json
INPUT		?= data/input/function_calling_tests.json
OUTPUT		?= data/output/function_calling_results.json
PYTHON		= @uv run python
ARGS		= --functions_definition $(FUNCTIONS) --input $(INPUT) --output $(OUTPUT)

.PHONY: install run debug clean lint lint-strict

install:
	@uv sync --no-cache

run:
	$(PYTHON) -m src $(ARGS)

debug:
	$(PYTHON) -m pdb -m src $(ARGS)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache

lint:
	@uv run flake8 .
	@uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	@uv run flake8 .
	@uv run mypy . --strict