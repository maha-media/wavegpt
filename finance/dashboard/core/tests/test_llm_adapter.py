from core.llm_adapter import get_adapter, BedrockAdapter, AnthropicAdapter, OpenAIAdapter


def test_get_adapter_bedrock(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    assert isinstance(get_adapter(), BedrockAdapter)


def test_get_adapter_anthropic(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert isinstance(get_adapter(), AnthropicAdapter)


def test_get_adapter_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert isinstance(get_adapter(), OpenAIAdapter)


def test_get_adapter_default():
    assert isinstance(get_adapter(), BedrockAdapter)
