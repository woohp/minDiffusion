.PHONY: format

python_files = *.py mindiffusion

format:
	isort $(python_files)
	black $(python_files)

lint:
	flake8 $(python_files)
	mypy $(python_files)
