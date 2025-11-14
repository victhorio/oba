run:
	@uv run oba

fmt:
	@uv run ruff format .
	@uv run ruff check --fix .

test:
	@uv run ruff check .
	@uv run pytest
