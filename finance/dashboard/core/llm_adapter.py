"""Pluggable LLM adapter — Bedrock (default), Anthropic, OpenAI."""

import json
import os
from abc import ABC, abstractmethod


SYSTEM_PROMPT = (
    "You are a trading assistant with full visibility into the user's live portfolio, "
    "regime classification, signal engine, and tax-aware rebalancing system. "
    "Answer questions about positions, P&L, regime, signals, tax optimization, "
    "and market conditions. Be concise. You cannot place orders."
)


def build_portfolio_context(balances, positions, orders, regime=None, portfolio=None):
    ctx = {
        "balances": balances,
        "positions": positions,
        "open_orders": orders,
    }
    if regime:
        ctx["regime"] = regime
    if portfolio:
        ctx["tax"] = portfolio.get("tax", {})
        ctx["tax_events"] = portfolio.get("tax_events", [])
    return json.dumps(ctx, indent=2, default=str)


class LLMAdapter(ABC):
    @abstractmethod
    async def stream_response(self, messages, portfolio_context):
        """Yield response text chunks."""
        ...


class BedrockAdapter(LLMAdapter):
    async def stream_response(self, messages, portfolio_context):
        import boto3
        client = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "system": f"{SYSTEM_PROMPT}\n\nCurrent portfolio:\n{portfolio_context}",
            "messages": messages,
        }
        response = client.invoke_model_with_response_stream(
            modelId=os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-opus-4-6-v1"),
            body=json.dumps(body),
        )
        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk["type"] == "content_block_delta":
                yield chunk["delta"].get("text", "")


class AnthropicAdapter(LLMAdapter):
    async def stream_response(self, messages, portfolio_context):
        import anthropic
        client = anthropic.AsyncAnthropic()
        async with client.messages.stream(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-6"),
            max_tokens=1024,
            system=f"{SYSTEM_PROMPT}\n\nCurrent portfolio:\n{portfolio_context}",
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text


class OpenAIAdapter(LLMAdapter):
    async def stream_response(self, messages, portfolio_context):
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        msgs = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nCurrent portfolio:\n{portfolio_context}"}]
        msgs.extend(messages)
        stream = await client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
            messages=msgs,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


_ADAPTERS = {
    "bedrock": BedrockAdapter,
    "anthropic": AnthropicAdapter,
    "openai": OpenAIAdapter,
}


def get_adapter():
    provider = os.environ.get("LLM_PROVIDER", "bedrock").lower()
    cls = _ADAPTERS.get(provider, BedrockAdapter)
    return cls()
