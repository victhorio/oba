import pytest

from .mock_openai import MockAsyncOpenAI


@pytest.fixture
def oai_client() -> MockAsyncOpenAI:
    return MockAsyncOpenAI()
