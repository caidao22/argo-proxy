import json
from asyncio import run
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from argoproxy.config import ArgoConfig
from argoproxy.endpoints import responses
from argoproxy.types.function_call import ResponseFunctionToolCall


def _make_config() -> ArgoConfig:
    return ArgoConfig(user="tester", _real_stream=False)


def _make_model_registry():
    return SimpleNamespace(
        resolve_model_name=lambda model_name, model_type=None: model_name,
        available_chat_models={"argo:gpt-4o": "gpt4o"},
        no_sys_msg_models={},
        option_2_input_models={},
        native_tool_call_models={},
    )


def test_reorder_tool_messages_places_tool_result_next_to_assistant_call():
    messages = [
        {"role": "user", "content": "Use the tool."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "user", "content": "interleaved"},
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
    ]

    reordered = responses._reorder_tool_messages(messages)

    assert [msg["role"] for msg in reordered] == ["user", "assistant", "tool", "user"]
    assert reordered[2]["tool_call_id"] == "call_1"


def test_reorder_tool_messages_rejects_unmatched_tool_message():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "tool", "tool_call_id": "call_missing", "content": "result"},
    ]

    with pytest.raises(ValueError, match="Unmatched tool messages"):
        responses._reorder_tool_messages(messages)


def test_prepare_request_data_reorders_tool_outputs_for_responses_input():
    request_data = {
        "model": "argo:gpt-4o",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Use tool"}]},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": "{}",
                "id": "fc_1",
            },
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "after"}]},
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": json.dumps({"ok": True}),
            },
        ],
        "stream": False,
    }

    prepared = responses.prepare_request_data(
        request_data,
        _make_config(),
        _make_model_registry(),
    )

    roles = [msg["role"] for msg in prepared["messages"]]
    assert roles == ["user", "assistant", "tool", "user"]
    assert prepared["messages"][2]["tool_call_id"] == "call_1"


def test_send_streaming_request_emits_function_call_argument_events(monkeypatch):
    sent_events = []

    async def fake_send_off_sse(response, data):
        sent_events.append(data)

    class FakeContent:
        async def iter_any(self):
            if False:
                yield b""

    class FakeUpstreamResponse:
        status = 200
        headers = {}
        content = FakeContent()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def json(self):
            return {"response": "tool call text"}

        async def text(self):
            return json.dumps({"response": "tool call text"})

    class FakeSession:
        def post(self, *args, **kwargs):
            return FakeUpstreamResponse()

    class FakeStreamResponse:
        def __init__(self, status=None, headers=None):
            self.status = status
            self.headers = headers or {}

        def enable_chunked_encoding(self):
            return None

        async def prepare(self, request):
            return None

        async def write(self, chunk):
            return None

        async def write_eof(self):
            return None

    monkeypatch.setattr(responses, "send_off_sse", fake_send_off_sse)
    monkeypatch.setattr(responses.web, "StreamResponse", FakeStreamResponse)
    monkeypatch.setattr(responses, "calculate_prompt_tokens_async", AsyncMock(return_value=11))
    monkeypatch.setattr(responses, "calculate_completion_tokens_async", AsyncMock(return_value=7))
    monkeypatch.setattr(responses, "log_upstream_response", lambda *args, **kwargs: None)
    monkeypatch.setattr(responses, "determine_model_family", lambda model: "openai")

    fake_tool_call = ResponseFunctionToolCall(
        arguments='{"city":"Chicago"}',
        call_id="call_123",
        id="fc_123",
        name="lookup_weather",
        status="completed",
    )

    class FakeToolInterceptor:
        def process(self, response_text, model_family, request_data=None):
            return ([{"name": "lookup_weather", "arguments": {"city": "Chicago"}}], "")

    monkeypatch.setattr(responses, "ToolInterceptor", FakeToolInterceptor)
    monkeypatch.setattr(responses, "tool_calls_to_openai", lambda calls, api_format=None: [fake_tool_call])

    request = SimpleNamespace()
    result = run(
        responses.send_streaming_request(
            FakeSession(),
            _make_config(),
            {"model": "argo:gpt-4o", "tools": [{"type": "function", "name": "lookup_weather"}]},
            request,
            pseudo_stream=True,
        )
    )

    event_types = [event["type"] for event in sent_events]
    assert "response.function_call_arguments.delta" in event_types
    assert "response.function_call_arguments.done" in event_types

    delta_event = next(
        event for event in sent_events if event["type"] == "response.function_call_arguments.delta"
    )
    done_event = next(
        event for event in sent_events if event["type"] == "response.function_call_arguments.done"
    )
    completed_event = next(event for event in sent_events if event["type"] == "response.completed")

    assert delta_event["item_id"] == "fc_123"
    assert done_event["item_id"] == "fc_123"
    assert completed_event["response"]["usage"]["output_tokens"] == 7
    assert completed_event["response"]["output"][0]["type"] == "function_call"
    assert result.status == 200
