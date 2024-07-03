.PHONY: format

python_files = *.py mindiffusion

format:
	ruff check --fix-only $(python_files)
	ruff format $(python_files)

lint:
	ruff check $(python_files)
	mypy $(python_files)
