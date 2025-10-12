run:
	@uv run oba

check-and-fmt:
	ruff format .
	ruff check --fix .
