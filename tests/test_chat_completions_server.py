import asyncio
import base64
import json
from types import SimpleNamespace

import click
import httpx
from click.testing import CliRunner
import llm
import pytest
import sqlite_utils
from llm.logs import LogStore
from llm.parts import (
    AttachmentPart,
    Message,
    StreamEvent,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)

import llm_chat_completions_server


@pytest.fixture(autouse=True)
def isolated_llm_user_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "user_dir", lambda: tmp_path)
    return tmp_path


def request(app, method, path, **kwargs):
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(run())


class AsyncMockModel(llm.AsyncModel):
    model_id = "async-mock"
    can_stream = True
    supports_schema = True
    supports_tools = True

    class Options(llm.Options):
        temperature: float | None = None
        max_tokens: int | None = None
        top_p: float | None = None

    def __init__(self, events):
        self.events = events
        self.calls = []

    async def execute(self, prompt, stream, response, conversation):
        self.calls.append(
            {
                "prompt": prompt,
                "stream": stream,
                "response": response,
                "conversation": conversation,
            }
        )
        for event in self.events:
            yield event
        response.set_usage(
            input=3,
            output=2,
            details={"completion_tokens_details": {"reasoning_tokens": 1}},
        )


def use_model(monkeypatch, model):
    def get_async_model(model_id):
        if model_id in (model.model_id, "model-alias"):
            return model
        raise llm.UnknownModelError(f"Unknown model: {model_id}")

    monkeypatch.setattr(llm, "get_async_model", get_async_model)


def test_models_lists_only_async_models(monkeypatch):
    monkeypatch.setattr(
        llm,
        "get_async_models",
        lambda: [
            SimpleNamespace(model_id="async-one"),
            SimpleNamespace(model_id="async-two"),
        ],
    )

    response = request(llm_chat_completions_server.create_app(), "GET", "/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [
            {
                "id": "async-one",
                "object": "model",
                "created": 0,
                "owned_by": "llm",
            },
            {
                "id": "async-two",
                "object": "model",
                "created": 0,
                "owned_by": "llm",
            },
        ],
    }


def test_chat_completion_uses_async_model_and_llm_messages(monkeypatch):
    model = AsyncMockModel(["Hello", " there"])
    use_model(monkeypatch, model)

    response = request(
        llm_chat_completions_server.create_app(),
        "POST",
        "/v1/chat/completions",
        headers={"Authorization": "Bearer ignored"},
        json={
            "model": "model-alias",
            "messages": [
                {"role": "system", "content": "Be terse"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"q":"cats"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "A result",
                },
                {"role": "user", "content": "Now answer"},
            ],
            "temperature": 0.25,
            "max_tokens": 100,
            "top_p": 0.9,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["id"].startswith("chatcmpl-")
    assert data["object"] == "chat.completion"
    assert data["model"] == "async-mock"
    assert data["choices"] == [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello there",
                "refusal": None,
            },
            "logprobs": None,
            "finish_reason": "stop",
        }
    ]
    assert data["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
        "completion_tokens_details": {"reasoning_tokens": 1},
    }

    call = model.calls[0]
    assert call["stream"] is False
    assert call["conversation"] is None
    prompt = call["prompt"]
    assert prompt.options.temperature == 0.25
    assert prompt.options.max_tokens == 100
    assert prompt.options.top_p == 0.9
    expected_messages = [
        Message(role="system", parts=[TextPart(text="Be terse")]),
        Message(
            role="user",
            parts=[
                TextPart(text="Describe this"),
                AttachmentPart(
                    attachment=llm.Attachment(
                        type="image/png",
                        content=b"\x89PNG\r\n\x1a\n",
                    )
                ),
            ],
        ),
        Message(
            role="assistant",
            parts=[
                ToolCallPart(
                    name="lookup",
                    arguments={"q": "cats"},
                    tool_call_id="call_1",
                )
            ],
        ),
        Message(
            role="tool",
            parts=[
                ToolResultPart(
                    name="lookup",
                    output="A result",
                    tool_call_id="call_1",
                )
            ],
        ),
        Message(role="user", parts=[TextPart(text="Now answer")]),
    ]
    assert [message.to_dict() for message in prompt.messages] == [
        message.to_dict() for message in expected_messages
    ]


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
@pytest.mark.parametrize("capture", ["model", "logs"])
@pytest.mark.parametrize(
    "content",
    [
        pytest.param(
            ["Before", b"first image", "Between", b"second image", "After"],
            id="interleaved",
        ),
        pytest.param([b"first image", "After"], id="image-first"),
        pytest.param(["First", "Second"], id="adjacent-text"),
        pytest.param(["", b"first image"], id="empty-text"),
        pytest.param([""], id="empty-text-only"),
        pytest.param(["", ""], id="adjacent-empty-text"),
        pytest.param("Plain text", id="string"),
        pytest.param(["One text part"], id="text-array"),
        pytest.param(
            ["Describe", b"first image", b"second image"],
            id="text-then-images",
        ),
        pytest.param([b"first image", b"second image"], id="images-only"),
    ],
)
def test_trailing_user_content_preserves_canonical_input(
    isolated_llm_user_dir, monkeypatch, content, capture, stream
):
    # Build client content and its expected canonical value independently of
    # the server's conversion/splitting helpers. Data URLs keep logging offline.
    expected_parts = []
    client_parts = []
    for part in [content] if isinstance(content, str) else content:
        if isinstance(part, str):
            expected_parts.append(TextPart(text=part))
            client_parts.append({"type": "text", "text": part})
        else:
            expected_parts.append(
                AttachmentPart(
                    attachment=llm.Attachment(type="image/png", content=part)
                )
            )
            client_parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,"
                        + base64.b64encode(part).decode()
                    },
                }
            )
    expected_messages = [
        Message(role="system", parts=[TextPart(text="Keep order")]),
        Message(role="user", parts=[TextPart(text="Earlier question")]),
        Message(role="assistant", parts=[TextPart(text="Earlier answer")]),
        Message(role="user", parts=expected_parts),
    ]
    model = AsyncMockModel(["Done"])
    model.attachment_types = {"image/png"}
    use_model(monkeypatch, model)

    response = request(
        llm_chat_completions_server.create_app(),
        "POST",
        "/v1/chat/completions",
        json={
            "model": model.model_id,
            "messages": [
                {"role": "system", "content": "Keep order"},
                {"role": "user", "content": "Earlier question"},
                {"role": "assistant", "content": "Earlier answer"},
                {
                    "role": "user",
                    "content": content if isinstance(content, str) else client_parts,
                },
            ],
            "stream": stream,
        },
    )
    assert response.status_code == 200
    if stream:
        assert response.text.endswith("data: [DONE]\n\n")
        assert '"error"' not in response.text
    else:
        assert response.json()["choices"][0]["message"]["content"] == "Done"
    assert len(model.calls) == 1

    if capture == "model":
        prompt = model.calls[0]["prompt"]
        observed_messages = prompt.messages
    else:
        db = sqlite_utils.Database(isolated_llm_user_dir / "logs.db")
        turns = list(db["turns"].rows)
        assert len(turns) == 1
        observed_messages = LogStore(db).load_chain(turns[0]["parent_message_hash"])
    assert [message.to_dict() for message in observed_messages] == [
        message.to_dict() for message in expected_messages
    ]
    if capture == "model":
        assert prompt.system == "Keep order"
        assert prompt.prompt == "".join(
            part.text for part in expected_parts if isinstance(part, TextPart)
        )
        assert [attachment.content for attachment in prompt.attachments] == [
            part.attachment.content
            for part in expected_parts
            if isinstance(part, AttachmentPart)
        ]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("attachment_types", [set(), {"image/jpeg"}])
@pytest.mark.parametrize("image_first", [False, True])
def test_ordered_content_retains_attachment_rejection(
    isolated_llm_user_dir, monkeypatch, stream, attachment_types, image_first
):
    model = AsyncMockModel(["Must not execute"])
    model.attachment_types = attachment_types
    use_model(monkeypatch, model)
    text = {"type": "text", "text": "Describe"}
    image = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,eA=="},
    }
    response = request(
        llm_chat_completions_server.create_app(),
        "POST",
        "/v1/chat/completions",
        json={
            "model": model.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [image, text] if image_first else [text, image],
                }
            ],
            "stream": stream,
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert model.calls == []
    assert not (isolated_llm_user_dir / "logs.db").exists()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("content_array", [False, True])
def test_custom_prompt_override_is_preserved(monkeypatch, stream, content_array):
    class CustomPromptModel(AsyncMockModel):
        def prompt(self, *args, **kwargs):
            kwargs["prompt"] = (kwargs["prompt"] or "") + " transformed"
            response = super().prompt(*args, **kwargs)
            response.prompt.plugin_marker = "custom"
            self.created_response = response
            self.created_prompt = response.prompt
            return response

    model = CustomPromptModel(["Done"])
    use_model(monkeypatch, model)
    content = (
        [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]
        if content_array
        else "AB"
    )
    response = request(
        llm_chat_completions_server.create_app(),
        "POST",
        "/v1/chat/completions",
        json={
            "model": model.model_id,
            "messages": [{"role": "user", "content": content}],
            "stream": stream,
        },
    )
    assert response.status_code == 200
    assert len(model.calls) == 1
    assert model.calls[0]["response"] is model.created_response
    assert model.calls[0]["prompt"] is model.created_prompt
    assert model.created_prompt.plugin_marker == "custom"
    assert model.created_prompt.prompt == "AB transformed"


def test_chat_completion_passes_tools_and_returns_tool_calls(monkeypatch):
    model = AsyncMockModel(
        [
            StreamEvent(
                type="tool_call_name",
                chunk="get_weather",
                tool_call_id="call_weather",
            ),
            StreamEvent(
                type="tool_call_args",
                chunk='{"city":"Paris"}',
                tool_call_id="call_weather",
            ),
        ]
    )
    use_model(monkeypatch, model)

    response = request(
        llm_chat_completions_server.create_app(),
        "POST",
        "/v1/chat/completions",
        json={
            "model": "async-mock",
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0] == {
        "index": 0,
        "message": {
            "role": "assistant",
            "content": None,
            "refusal": None,
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Paris"}',
                    },
                }
            ],
        },
        "logprobs": None,
        "finish_reason": "tool_calls",
    }
    tool = model.calls[0]["prompt"].tools[0]
    assert tool.name == "get_weather"
    assert tool.description == "Get the weather"
    assert tool.input_schema["required"] == ["city"]
    assert tool.implementation is None


def test_streaming_chat_completion(monkeypatch):
    model = AsyncMockModel(
        [
            StreamEvent(type="text", chunk="Hel"),
            StreamEvent(type="text", chunk="lo"),
        ]
    )
    use_model(monkeypatch, model)

    response = request(
        llm_chat_completions_server.create_app(),
        "POST",
        "/v1/chat/completions",
        json={
            "model": "async-mock",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    lines = [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert lines[-1] == "[DONE]"
    chunks = [json.loads(line) for line in lines[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {
        "role": "assistant",
        "content": "",
    }
    assert [chunk["choices"][0]["delta"] for chunk in chunks[1:3]] == [
        {"content": "Hel"},
        {"content": "lo"},
    ]
    assert chunks[3]["choices"][0]["finish_reason"] == "stop"
    assert chunks[4]["choices"] == []
    assert chunks[4]["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
        "completion_tokens_details": {"reasoning_tokens": 1},
    }
    assert model.calls[0]["stream"] is True


def test_completed_responses_are_logged(isolated_llm_user_dir, monkeypatch):
    model = AsyncMockModel(["Logged response"])
    use_model(monkeypatch, model)

    response = request(
        llm_chat_completions_server.create_app(),
        "POST",
        "/v1/chat/completions",
        json={
            "model": "async-mock",
            "messages": [
                {"role": "system", "content": "System instructions"},
                {"role": "user", "content": "Log this prompt"},
            ],
        },
    )

    assert response.status_code == 200
    db = sqlite_utils.Database(isolated_llm_user_dir / "logs.db")
    turns = list(db["turns"].rows)
    assert len(turns) == 1
    assert turns[0]["model"] == "async-mock"
    search = list(db["turn_search"].rows)
    assert len(search) == 1
    assert search[0]["prompt"] == "Log this prompt"
    assert search[0]["response"] == "Logged response"
    assert db["messages"].count == 3
    assert {row["role"] for row in db["messages"].rows} == {
        "system",
        "user",
        "assistant",
    }


def test_streamed_responses_are_logged(isolated_llm_user_dir, monkeypatch):
    model = AsyncMockModel(
        [
            StreamEvent(type="text", chunk="Streamed "),
            StreamEvent(type="text", chunk="response"),
        ]
    )
    use_model(monkeypatch, model)

    response = request(
        llm_chat_completions_server.create_app(),
        "POST",
        "/v1/chat/completions",
        json={
            "model": "async-mock",
            "messages": [{"role": "user", "content": "Stream this"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    db = sqlite_utils.Database(isolated_llm_user_dir / "logs.db")
    turns = list(db["turns"].rows)
    assert len(turns) == 1
    search = list(db["turn_search"].rows)
    assert len(search) == 1
    assert search[0]["prompt"] == "Stream this"
    assert search[0]["response"] == "Streamed response"


def test_unknown_or_sync_only_model_returns_openai_error(monkeypatch):
    def get_async_model(model_id):
        raise llm.UnknownModelError(
            f"Unknown async model (sync model exists): {model_id}"
        )

    monkeypatch.setattr(llm, "get_async_model", get_async_model)

    response = request(
        llm_chat_completions_server.create_app(),
        "POST",
        "/v1/chat/completions",
        json={
            "model": "sync-only",
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "message": "The model 'sync-only' does not exist or is not available asynchronously.",
            "type": "invalid_request_error",
            "param": "model",
            "code": "model_not_found",
        }
    }


def test_invalid_request_returns_openai_error(monkeypatch):
    model = AsyncMockModel(["unused"])
    use_model(monkeypatch, model)

    response = request(
        llm_chat_completions_server.create_app(),
        "POST",
        "/v1/chat/completions",
        json={"model": "async-mock", "messages": "not-a-list"},
    )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "message": "'messages' must be an array.",
        "type": "invalid_request_error",
        "param": "messages",
        "code": None,
    }
    assert model.calls == []


def test_command_defaults_and_reload(monkeypatch):
    calls = []
    monkeypatch.setattr(
        llm_chat_completions_server.uvicorn,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    cli = click.Group()
    llm_chat_completions_server.register_commands(cli)

    result = CliRunner().invoke(cli, ["chat-completions-server", "--reload"])

    assert result.exit_code == 0
    assert calls == [
        (
            ("llm_chat_completions_server:app",),
            {"host": "127.0.0.1", "port": 8002, "reload": True},
        )
    ]


def test_command_custom_port(monkeypatch):
    calls = []
    monkeypatch.setattr(
        llm_chat_completions_server.uvicorn,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    cli = click.Group()
    llm_chat_completions_server.register_commands(cli)

    result = CliRunner().invoke(cli, ["chat-completions-server", "-p", "8123"])

    assert result.exit_code == 0
    assert calls[0][1]["port"] == 8123
    assert calls[0][1]["reload"] is False
