run:
	@uv run oba

fmt:
	@uv run ruff format .
	@uv run ruff check --fix .

test:
	@uv run ruff check .
	@uv run pytest

test-manual:
	@uv run python -c "import asyncio; from ag import run_manual_tests; asyncio.run(run_manual_tests())"
	@uv run python -c "import asyncio; from ag.models import run_manual_tests; asyncio.run(run_manual_tests())"
