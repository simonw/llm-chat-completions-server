import asyncio
import base64
import binascii
import json
import mimetypes
import time
from urllib.parse import unquote_to_bytes, urlparse

import click
import llm
import sqlite_utils
import uvicorn
from llm.parts import (
    AttachmentPart,
    StreamEvent,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route


class RequestError(Exception):
    def __init__(self, message, param=None, code=None):
        super().__init__(message)
        self.message = message
        self.param = param
        self.code = code


def error_response(
    message,
    *,
    status_code=400,
    error_type="invalid_request_error",
    param=None,
    code=None,
):
    return JSONResponse(
        {
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
        status_code=status_code,
    )


async def list_models(request):
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": model.model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "llm",
                }
                for model in llm.get_async_models()
            ],
        }
    )


async def chat_completions(request):
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return error_response(
            "Could not parse the request body as JSON.",
            code="invalid_json",
        )

    if not isinstance(body, dict):
        return error_response("The request body must be a JSON object.")

    try:
        model_id = body.get("model")
        if not isinstance(model_id, str) or not model_id:
            raise RequestError("'model' must be a non-empty string.", "model")

        messages_value = body.get("messages")
        if not isinstance(messages_value, list):
            raise RequestError("'messages' must be an array.", "messages")
        if not messages_value:
            raise RequestError("'messages' must not be empty.", "messages")

        n = body.get("n", 1)
        if n != 1:
            raise RequestError(
                "Only n=1 is supported.",
                "n",
                "unsupported_value",
            )

        stream = body.get("stream", False)
        if not isinstance(stream, bool):
            raise RequestError("'stream' must be a boolean.", "stream")

        try:
            model = llm.get_async_model(model_id)
        except llm.UnknownModelError:
            return error_response(
                f"The model '{model_id}' does not exist or is not available asynchronously.",
                status_code=404,
                param="model",
                code="model_not_found",
            )

        messages = openai_messages_to_llm(messages_value)
        tools = openai_tools_to_llm(body)
        options, schema = options_and_schema(model, body)
        legacy_system = system_from_messages(messages)
        messages, prompt, attachments = split_final_user_message(messages)

        response = model.prompt(
            prompt=prompt,
            system=legacy_system,
            attachments=attachments,
            messages=messages,
            stream=stream,
            tools=tools,
            schema=schema,
            options=options,
        )
    except RequestError as ex:
        return error_response(ex.message, param=ex.param, code=ex.code)
    except ValueError as ex:
        return error_response(str(ex))

    completion_id = f"chatcmpl-{response.id}"
    created = int(time.time())
    response_model = response.resolved_model or model.model_id

    if stream:
        include_usage = bool(
            isinstance(body.get("stream_options"), dict)
            and body["stream_options"].get("include_usage")
        )
        return StreamingResponse(
            stream_response(
                response,
                completion_id=completion_id,
                created=created,
                model=response_model,
                include_usage=include_usage,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        output_messages = await response.messages()
        usage = await response.usage()
        await log_response_to_db(response)
    except Exception as ex:
        return model_error_response(ex)

    message, finish_reason = completion_message(output_messages)
    return JSONResponse(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": response.resolved_model or model.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage_dict(usage),
        }
    )


def openai_messages_to_llm(messages):
    converted = []
    tool_names = {}
    for index, message in enumerate(messages):
        param = f"messages[{index}]"
        if not isinstance(message, dict):
            raise RequestError(f"'{param}' must be an object.", param)

        role = message.get("role")
        if not isinstance(role, str) or role not in {
            "system",
            "developer",
            "user",
            "assistant",
            "tool",
        }:
            raise RequestError(
                f"'{param}.role' must be one of system, developer, user, assistant, or tool.",
                f"{param}.role",
            )
        llm_role = "system" if role == "developer" else role
        parts = content_to_parts(message.get("content"), f"{param}.content")

        if role == "assistant":
            tool_calls = message.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                raise RequestError(
                    f"'{param}.tool_calls' must be an array.",
                    f"{param}.tool_calls",
                )
            for tool_index, tool_call in enumerate(tool_calls):
                tool_param = f"{param}.tool_calls[{tool_index}]"
                if not isinstance(tool_call, dict):
                    raise RequestError(f"'{tool_param}' must be an object.", tool_param)
                if tool_call.get("type", "function") != "function":
                    raise RequestError(
                        f"'{tool_param}.type' must be 'function'.",
                        f"{tool_param}.type",
                    )
                function = tool_call.get("function")
                if not isinstance(function, dict):
                    raise RequestError(
                        f"'{tool_param}.function' must be an object.",
                        f"{tool_param}.function",
                    )
                name = function.get("name")
                if not isinstance(name, str) or not name:
                    raise RequestError(
                        f"'{tool_param}.function.name' must be a non-empty string.",
                        f"{tool_param}.function.name",
                    )
                tool_call_id = tool_call.get("id")
                if not isinstance(tool_call_id, str) or not tool_call_id:
                    raise RequestError(
                        f"'{tool_param}.id' must be a non-empty string.",
                        f"{tool_param}.id",
                    )
                arguments = function.get("arguments", "{}")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments or "{}")
                    except json.JSONDecodeError:
                        raise RequestError(
                            f"'{tool_param}.function.arguments' must be valid JSON.",
                            f"{tool_param}.function.arguments",
                        )
                if not isinstance(arguments, dict):
                    raise RequestError(
                        f"'{tool_param}.function.arguments' must decode to an object.",
                        f"{tool_param}.function.arguments",
                    )
                tool_names[tool_call_id] = name
                parts.append(
                    ToolCallPart(
                        name=name,
                        arguments=arguments,
                        tool_call_id=tool_call_id,
                    )
                )

        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise RequestError(
                    f"'{param}.tool_call_id' must be a non-empty string.",
                    f"{param}.tool_call_id",
                )
            text = "".join(part.text for part in parts if isinstance(part, TextPart))
            parts = [
                ToolResultPart(
                    name=message.get("name") or tool_names.get(tool_call_id) or "",
                    output=text,
                    tool_call_id=tool_call_id,
                )
            ]

        converted.append(llm.Message(role=llm_role, parts=parts))
    return converted


def content_to_parts(content, param):
    if content is None:
        return []
    if isinstance(content, str):
        return [TextPart(text=content)]
    if not isinstance(content, list):
        raise RequestError(f"'{param}' must be a string, an array, or null.", param)

    parts = []
    for index, content_part in enumerate(content):
        part_param = f"{param}[{index}]"
        if not isinstance(content_part, dict):
            raise RequestError(f"'{part_param}' must be an object.", part_param)
        part_type = content_part.get("type")
        if part_type == "text":
            text = content_part.get("text")
            if not isinstance(text, str):
                raise RequestError(
                    f"'{part_param}.text' must be a string.",
                    f"{part_param}.text",
                )
            parts.append(TextPart(text=text))
        elif part_type == "image_url":
            image_url = content_part.get("image_url")
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            if not isinstance(image_url, str) or not image_url:
                raise RequestError(
                    f"'{part_param}.image_url.url' must be a non-empty string.",
                    f"{part_param}.image_url.url",
                )
            parts.append(AttachmentPart(attachment=attachment_from_url(image_url)))
        else:
            raise RequestError(
                f"Unsupported content type {part_type!r} in '{part_param}'.",
                f"{part_param}.type",
            )
    return parts


def system_from_messages(messages):
    system_parts = []
    for message in messages:
        text = "".join(
            part.text for part in message.parts if isinstance(part, TextPart)
        )
        if message.role == "system" and text:
            system_parts.append(text)
    return "\n\n".join(system_parts) or None


def split_final_user_message(messages):
    """Split a trailing user message into (history, prompt, attachments).

    llm's model.prompt() treats messages= as prior history and appends
    prompt=/attachments= to it as this turn's new user message, so the
    final user message must move out of the list - left in both places
    it would be sent to the model and logged twice.
    """
    if messages and messages[-1].role == "user" and messages[-1].parts:
        final = messages[-1]
        prompt = "".join(
            part.text for part in final.parts if isinstance(part, TextPart)
        )
        attachments = [
            part.attachment for part in final.parts if isinstance(part, AttachmentPart)
        ]
        return messages[:-1], prompt or None, attachments
    return messages, None, []


def attachment_from_url(url):
    if url.startswith("data:"):
        try:
            header, encoded = url.split(",", 1)
            metadata = header[5:].split(";")
            media_type = metadata[0] or "application/octet-stream"
            if "base64" in metadata[1:]:
                content = base64.b64decode(encoded, validate=True)
            else:
                content = unquote_to_bytes(encoded)
        except (ValueError, binascii.Error):
            raise RequestError(
                "Image data URL is invalid.",
                "messages",
                "invalid_image",
            )
        return llm.Attachment(type=media_type, content=content)

    path = urlparse(url).path
    media_type, _ = mimetypes.guess_type(path)
    return llm.Attachment(type=media_type, url=url)


def openai_tools_to_llm(body):
    tools_value = body.get("tools")
    if tools_value is None and body.get("functions") is not None:
        functions = body["functions"]
        if not isinstance(functions, list):
            raise RequestError("'functions' must be an array.", "functions")
        tools_value = [
            {"type": "function", "function": function} for function in functions
        ]
    if tools_value is None:
        return []
    if not isinstance(tools_value, list):
        raise RequestError("'tools' must be an array.", "tools")

    tools = []
    for index, tool_value in enumerate(tools_value):
        param = f"tools[{index}]"
        if not isinstance(tool_value, dict):
            raise RequestError(f"'{param}' must be an object.", param)
        if tool_value.get("type") != "function":
            raise RequestError(
                f"'{param}.type' must be 'function'.",
                f"{param}.type",
            )
        function = tool_value.get("function")
        if not isinstance(function, dict):
            raise RequestError(
                f"'{param}.function' must be an object.",
                f"{param}.function",
            )
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise RequestError(
                f"'{param}.function.name' must be a non-empty string.",
                f"{param}.function.name",
            )
        parameters = function.get("parameters") or {}
        if not isinstance(parameters, dict):
            raise RequestError(
                f"'{param}.function.parameters' must be an object.",
                f"{param}.function.parameters",
            )
        tools.append(
            llm.Tool(
                name=name,
                description=function.get("description"),
                input_schema=parameters,
            )
        )
    return tools


def options_and_schema(model, body):
    model_fields = model.Options.model_fields
    options = {}
    for key in (
        "temperature",
        "max_tokens",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "logit_bias",
        "seed",
        "logprobs",
        "top_logprobs",
        "parallel_tool_calls",
        "tool_choice",
    ):
        if key in body and key in model_fields:
            options[key] = body[key]

    if (
        "max_completion_tokens" in body
        and "max_tokens" not in options
        and "max_tokens" in model_fields
    ):
        options["max_tokens"] = body["max_completion_tokens"]

    schema = None
    response_format = body.get("response_format")
    if response_format is not None:
        if not isinstance(response_format, dict):
            raise RequestError(
                "'response_format' must be an object.",
                "response_format",
            )
        response_type = response_format.get("type")
        if response_type == "json_object":
            if "json_object" in model_fields:
                options["json_object"] = True
        elif response_type == "json_schema":
            json_schema = response_format.get("json_schema")
            if not isinstance(json_schema, dict) or not isinstance(
                json_schema.get("schema"), dict
            ):
                raise RequestError(
                    "'response_format.json_schema.schema' must be an object.",
                    "response_format.json_schema.schema",
                )
            schema = json_schema["schema"]
        elif response_type != "text":
            raise RequestError(
                f"Unsupported response format {response_type!r}.",
                "response_format.type",
            )
    return options, schema


def completion_message(messages):
    parts = [part for message in messages for part in message.parts]
    content = "".join(part.text for part in parts if isinstance(part, TextPart))
    tool_calls = [
        {
            "id": part.tool_call_id,
            "type": "function",
            "function": {
                "name": part.name,
                "arguments": json.dumps(
                    part.arguments,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
            },
        }
        for part in parts
        if isinstance(part, ToolCallPart)
    ]
    message = {
        "role": "assistant",
        "content": content or None,
        "refusal": None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message, "tool_calls" if tool_calls else "stop"


def usage_dict(usage):
    prompt_tokens = usage.input or 0
    completion_tokens = usage.output or 0
    result = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if usage.details:
        result.update(usage.details)
    return result


def chunk_dict(completion_id, created, model, choices, usage=None):
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": choices,
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def sse(data):
    return "data: {}\n\n".format(
        json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    )


async def stream_response(
    response,
    *,
    completion_id,
    created,
    model,
    include_usage,
):
    yield sse(
        chunk_dict(
            completion_id,
            created,
            model,
            [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
        )
    )

    tool_indexes = {}
    saw_tool_call = False
    try:
        async for event in response.astream_events():
            if not isinstance(event, StreamEvent):
                continue
            delta = None
            if event.type == "text":
                delta = {"content": event.chunk}
            elif event.type in {"tool_call_name", "tool_call_args"}:
                saw_tool_call = True
                key = (
                    event.message_index,
                    event.part_index,
                    event.tool_call_id,
                )
                if key not in tool_indexes:
                    tool_indexes[key] = len(tool_indexes)
                tool_index = tool_indexes[key]
                if event.type == "tool_call_name":
                    tool_call = {
                        "index": tool_index,
                        "id": event.tool_call_id or f"call_{tool_index}",
                        "type": "function",
                        "function": {
                            "name": event.chunk,
                            "arguments": "",
                        },
                    }
                else:
                    tool_call = {
                        "index": tool_index,
                        "function": {"arguments": event.chunk},
                    }
                delta = {"tool_calls": [tool_call]}
            elif event.type == "reasoning":
                # Classic Chat Completions has no portable reasoning field.
                continue

            if delta is not None:
                yield sse(
                    chunk_dict(
                        completion_id,
                        created,
                        response.resolved_model or model,
                        [
                            {
                                "index": 0,
                                "delta": delta,
                                "logprobs": None,
                                "finish_reason": None,
                            }
                        ],
                    )
                )

        await log_response_to_db(response)
        yield sse(
            chunk_dict(
                completion_id,
                created,
                response.resolved_model or model,
                [
                    {
                        "index": 0,
                        "delta": {},
                        "logprobs": None,
                        "finish_reason": "tool_calls" if saw_tool_call else "stop",
                    }
                ],
            )
        )
        if include_usage:
            usage = await response.usage()
            yield sse(
                chunk_dict(
                    completion_id,
                    created,
                    response.resolved_model or model,
                    [],
                    usage=usage_dict(usage),
                )
            )
    except Exception as ex:
        yield sse(model_error(ex))
    yield "data: [DONE]\n\n"


def model_error(ex):
    if isinstance(ex, llm.NeedsKeyException):
        message = f"Server model credentials are not configured: {ex}"
    else:
        message = str(ex) or ex.__class__.__name__
    return {
        "error": {
            "message": message,
            "type": "server_error",
            "param": None,
            "code": None,
        }
    }


def model_error_response(ex):
    return JSONResponse(model_error(ex), status_code=500)


async def log_response_to_db(response):
    user_dir = llm.user_dir()
    if (user_dir / "logs-off").exists():
        return
    sync_response = await response.to_sync_response()

    def write_log():
        sync_response.log_to_db(sqlite_utils.Database(user_dir / "logs.db"))

    await asyncio.to_thread(write_log)


def create_app():
    return Starlette(
        routes=[
            Route("/v1/models", list_models, methods=["GET"]),
            Route(
                "/v1/chat/completions",
                chat_completions,
                methods=["POST"],
            ),
        ]
    )


app = create_app()


@llm.hookimpl
def register_commands(cli):
    @cli.command(name="chat-completions-server")
    @click.option(
        "-p",
        "--port",
        type=click.IntRange(1, 65535),
        default=8002,
        show_default=True,
        help="Port to listen on.",
    )
    @click.option(
        "--reload",
        is_flag=True,
        help="Restart the development server when Python files change.",
    )
    def chat_completions_server(port, reload):
        "Serve an OpenAI-compatible API using LLM async models."
        uvicorn.run(
            "llm_chat_completions_server:app",
            host="127.0.0.1",
            port=port,
            reload=reload,
        )
