import asyncio
import json
import time
import uuid
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Union

import aiohttp
from aiohttp import web

from ..config import ArgoConfig
from ..models import ModelRegistry
from ..tool_calls.output_handle import ToolInterceptor, tool_calls_to_openai
from ..types import (
    Response,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from ..types.responses import (
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
)
from ..utils.logging import (
    log_converted_request,
    log_error,
    log_original_request,
    log_upstream_error,
    log_upstream_response,
)
from ..utils.misc import apply_username_passthrough
from ..utils.models import apply_claude_max_tokens_limit, determine_model_family
from ..utils.tokens import (
    calculate_prompt_tokens_async,
    count_tokens,
    count_tokens_async,
)
from ..utils.stream_decoder import StreamDecoder
from ..utils.transports import send_off_sse
from ..utils.usage import calculate_completion_tokens_async, create_usage
from .chat import (
    prepare_chat_request_data,
    send_non_streaming_request,
)

INCOMPATIBLE_INPUT_FIELDS = {
    "include",
    "metadata",
    "previous_response_id",
    "reasoning",
    "service_tier",
    "store",
    "text",
    "truncation",
}


def transform_non_streaming_response(
    content: str,
    *,
    model_name: str,
    create_timestamp: int,
    prompt_tokens: int,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Transforms a non-streaming custom API response into a format compatible with OpenAI's API.

    Args:
        content: The response obtained from the custom API.
        model_name: The name of the model that generated the completion.
        create_timestamp: The creation timestamp of the completion.
        prompt_tokens: The number of tokens in the input prompt.

    Returns:
        A dictionary representing the OpenAI-compatible JSON response.
    """
    try:
        # Note: using sync count_tokens here as the original function was sync.
        # However, our unified calculate_completion_tokens_async is async.
        # Since transform_non_streaming_response is sync, we'll keep it as is or
        # just use the async version in the async counterpart.
        completion_tokens = count_tokens(content, model_name)
        if tool_calls:
            if (
                tool_calls
                and len(tool_calls) > 0
                and hasattr(tool_calls[0], "serialize")
            ):
                serializable_tool_calls = [
                    tc.serialize("openai-response") for tc in tool_calls
                ]
            else:
                serializable_tool_calls = tool_calls
            tool_tokens = count_tokens(json.dumps(serializable_tool_calls), model_name)
            completion_tokens += tool_tokens
        usage = create_usage(prompt_tokens, completion_tokens, api_type="response")

        output = []
        if tool_calls:
            output.extend(tool_calls_to_openai(tool_calls, api_format="response"))
        output.append(
            ResponseOutputMessage(
                id=f"msg_{uuid.uuid4().hex}",
                status="completed",
                content=[
                    ResponseOutputText(
                        text=content,
                    )
                ],
            )
        )

        openai_response = Response(
            id=f"resp_{uuid.uuid4().hex}",
            created_at=create_timestamp,
            model=model_name,
            output=output,
            status="completed",
            usage=usage,
        )

        return openai_response.model_dump()

    except json.JSONDecodeError as err:
        log_error(
            f"Error decoding JSON: {err}", context="responses.transform_non_streaming"
        )
        return {"error": f"Error decoding JSON: {err}"}
    except Exception as err:
        log_error(
            f"An error occurred: {err}", context="responses.transform_non_streaming"
        )
        return {"error": f"An error occurred: {err}"}


async def transform_non_streaming_response_async(
    content: str,
    *,
    model_name: str,
    create_timestamp: int,
    prompt_tokens: int,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Asynchronously transforms a non-streaming custom API response into a format compatible with OpenAI's API.

    Args:
        content: The response obtained from the custom API.
        model_name: The name of the model that generated the completion.
        create_timestamp: The creation timestamp of the completion.
        prompt_tokens: The number of tokens in the input prompt.

    Returns:
        A dictionary representing the OpenAI-compatible JSON response.
    """
    try:
        completion_tokens = await calculate_completion_tokens_async(
            content, tool_calls, model_name, api_format="response"
        )
        usage = create_usage(prompt_tokens, completion_tokens, api_type="response")

        output = []
        if tool_calls:
            output.extend(tool_calls_to_openai(tool_calls, api_format="response"))
        output.append(
            ResponseOutputMessage(
                id=f"msg_{uuid.uuid4().hex}",
                status="completed",
                content=[
                    ResponseOutputText(
                        text=content,
                    )
                ],
            )
        )

        openai_response = Response(
            id=f"resp_{uuid.uuid4().hex}",
            created_at=create_timestamp,
            model=model_name,
            output=output,
            status="completed",
            usage=usage,
        )

        return openai_response.model_dump()

    except json.JSONDecodeError as err:
        log_error(
            f"Error decoding JSON: {err}",
            context="responses.transform_non_streaming_async",
        )
        return {"error": f"Error decoding JSON: {err}"}
    except Exception as err:
        log_error(
            f"An error occurred: {err}",
            context="responses.transform_non_streaming_async",
        )
        return {"error": f"An error occurred: {err}"}


def transform_streaming_response(
    custom_response: Any,
    **kwargs,
) -> Dict[str, Any]:
    """
    Transforms a streaming custom API response into a format compatible with OpenAI's API.

    Args:
        custom_response: The response obtained from the custom API.
        model_name: The name of the model that generated the completion.

    Returns:
        A dictionary representing the OpenAI-compatible JSON response.
    """
    try:
        if isinstance(custom_response, str):
            custom_response_dict = json.loads(custom_response)
        else:
            custom_response_dict = custom_response

        response_text = custom_response_dict.get("response", "")
        content_index = kwargs.get("content_index", 0)
        output_index = kwargs.get("output_index", 0)
        sequence_number = kwargs.get("sequence_number", 0)
        id = kwargs.get("id", f"msg_{str(uuid.uuid4().hex)}")

        openai_response = ResponseTextDeltaEvent(
            content_index=content_index,
            delta=response_text,
            item_id=id,
            output_index=output_index,
            sequence_number=sequence_number,
        )

        return openai_response.model_dump()

    except json.JSONDecodeError as err:
        log_error(
            f"Error decoding JSON: {err}", context="responses.transform_streaming"
        )
        return {"error": f"Error decoding JSON: {err}"}
    except Exception as err:
        log_error(f"An error occurred: {err}", context="responses.transform_streaming")
        return {"error": f"An error occurred: {err}"}


def _convert_responses_messages(
    items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert Responses API input items to Chat Completions messages format.

    Handles:
    - ``type: "message"`` → standard CC message (strip type, map roles, convert content blocks)
    - ``type: "function_call"`` → merged into preceding assistant message as ``tool_calls``
    - ``type: "function_call_output"`` → ``role: "tool"`` message
    """
    converted: List[Dict[str, Any]] = []

    for item in items:
        item_type = item.get("type")

        if item_type == "message":
            converted.append(_convert_message_item(item))

        elif item_type == "function_call":
            # Build a Chat Completions tool_call entry
            tool_call = {
                "id": item.get("call_id", ""),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "{}"),
                },
            }
            # Merge into the preceding assistant message if possible
            if converted and converted[-1].get("role") == "assistant":
                converted[-1].setdefault("tool_calls", []).append(tool_call)
            else:
                # No preceding assistant message — create one
                converted.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call],
                })

        elif item_type == "function_call_output":
            converted.append({
                "role": "tool",
                "tool_call_id": item.get("call_id", ""),
                "content": item.get("output", ""),
            })

        else:
            # Fallback: strip type and pass through
            out = {k: v for k, v in item.items() if k != "type"}
            if isinstance(out.get("content"), list):
                out["content"] = [
                    _convert_content_block(block) for block in out["content"]
                ]
            converted.append(out)

    # Reorder: ensure every tool message sits immediately after the
    # assistant message whose tool_calls it answers.  The Chat Completions
    # API requires this adjacency.
    return _reorder_tool_messages(converted)


def _reorder_tool_messages(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Reorder messages so tool responses sit right after their assistant.

    The Chat Completions API requires that every ``role: "tool"`` message
    immediately follows the assistant message whose ``tool_calls`` it
    answers.  After the Responses→CC conversion the ordering may be
    different (e.g. Codex may interleave ``function_call_output`` items
    with other items).

    Algorithm:
    1. Pull all tool messages out into a dict keyed by ``tool_call_id``.
    2. Walk the non-tool messages.  After each assistant message that has
       ``tool_calls``, insert the matching tool messages in order.
    3. Fail fast if any tool messages remain unmatched. Forwarding them would
       still violate the strict Chat Completions ordering contract.
    """
    # Separate tool messages from everything else
    tool_msgs: Dict[str, Dict[str, Any]] = {}  # tool_call_id → message
    other_msgs: List[Dict[str, Any]] = []

    for msg in messages:
        if msg.get("role") == "tool" and "tool_call_id" in msg:
            tool_msgs[msg["tool_call_id"]] = msg
        else:
            other_msgs.append(msg)

    # Rebuild with tool messages placed after their assistant
    reordered: List[Dict[str, Any]] = []
    used_tool_ids: set = set()

    for msg in other_msgs:
        reordered.append(msg)
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id", "")
                if tc_id in tool_msgs:
                    reordered.append(tool_msgs[tc_id])
                    used_tool_ids.add(tc_id)

    unmatched_tool_ids = [tc_id for tc_id in tool_msgs if tc_id not in used_tool_ids]
    if unmatched_tool_ids:
        raise ValueError(
            "Unmatched tool messages for tool_call_id(s): "
            f"{', '.join(unmatched_tool_ids)}"
        )

    return reordered


def _convert_message_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single Responses API message item to Chat Completions format."""
    out = {k: v for k, v in item.items() if k != "type"}

    # developer -> system
    if out.get("role") == "developer":
        out["role"] = "system"

    # Convert content blocks
    if isinstance(out.get("content"), list):
        out["content"] = [
            _convert_content_block(block) for block in out["content"]
        ]

    return out


def _convert_content_block(block: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single Responses API content block to Chat Completions format."""
    if not isinstance(block, dict):
        return block
    block_type = block.get("type")
    if block_type in ("input_text", "output_text"):
        return {"type": "text", "text": block.get("text", "")}
    if block_type == "input_image":
        # Convert to Chat Completions image_url format
        image_data = block.get("image_url") or block.get("source", {})
        return {"type": "image_url", "image_url": image_data}
    if block_type == "input_audio":
        audio_data = block.get("input_audio") or block.get("source", {})
        return {"type": "input_audio", "input_audio": audio_data}
    return block


def prepare_request_data(
    data: Dict[str, Any], config: ArgoConfig, model_registry: ModelRegistry
) -> Dict[str, Any]:
    """
    Prepares the incoming request data for response models.

    Args:
        data: The original request data.
        config: Application configuration.
        model_registry: The ModelRegistry object containing model mappings.

    Returns:
        The modified and prepared request data.
    """
    # Convert Responses API input to Chat Completions messages format
    raw_input = data.get("input", [])

    # OpenAI Responses API allows input to be a plain string
    if isinstance(raw_input, str):
        messages = [{"role": "user", "content": raw_input}]
    else:
        messages = _convert_responses_messages(raw_input)

    if instructions := data.get("instructions", ""):
        messages.insert(0, {"role": "system", "content": instructions})
        del data["instructions"]
    data["messages"] = messages
    del data["input"]

    if max_tokens := data.get("max_output_tokens", None):
        data["max_tokens"] = max_tokens
        del data["max_output_tokens"]

    # Convert Responses API tools format to Chat Completions format
    # Responses API: {type: "function", name, description, parameters}
    # Chat Completions: {type: "function", function: {name, description, parameters}}
    if "tools" in data and isinstance(data["tools"], list):
        cc_tools = []
        for tool in data["tools"]:
            if tool.get("type") == "function" and "function" not in tool:
                # Responses API format → Chat Completions format
                cc_tools.append({
                    "type": "function",
                    "function": {
                        k: v for k, v in tool.items() if k != "type"
                    },
                })
            elif tool.get("type") == "function" and "function" in tool:
                # Already in Chat Completions format
                cc_tools.append(tool)
            # Skip non-function tools (web_search_preview, computer_use_preview, etc.)
            # as the upstream Chat Completions API doesn't support them
        data["tools"] = cc_tools if cc_tools else None
        if not data["tools"]:
            del data["tools"]

    # Use shared chat request preparation logic
    data = prepare_chat_request_data(data, config, model_registry, enable_tools=True)

    # Drop unsupported fields
    for key in list(data.keys()):
        if key in INCOMPATIBLE_INPUT_FIELDS:
            del data[key]

    return data


async def send_streaming_request(
    session: aiohttp.ClientSession,
    config: ArgoConfig,
    data: Dict[str, Any],
    request: web.Request,
    *,
    pseudo_stream: bool = False,
) -> web.StreamResponse:
    """Sends a streaming request to an API and streams the response to the client.

    Args:
        session: The client session for making the request.
        config: The configuration object containing the API URLs.
        data: The JSON payload of the request.
        request: The web request used for streaming responses.
        convert_to_openai: If True, converts the response to OpenAI format.
        pseudo_stream: If True, simulates streaming even if the upstream does not support it.
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/plain",
        "Accept-Encoding": "identity",
    }

    # Set response headers based on the mode
    response_headers = {"Content-Type": "text/event-stream"}
    created_timestamp = int(time.time())
    prompt_tokens = await calculate_prompt_tokens_async(data, data["model"])

    if pseudo_stream:
        # Note: data["stream"] is already set to False in proxy_request when pseudo_stream is True
        api_url = config.argo_url
    else:
        api_url = config.argo_stream_url

    async with session.post(api_url, headers=headers, json=data) as upstream_resp:
        if upstream_resp.status != 200:
            error_text = await upstream_resp.text()
            log_upstream_error(
                upstream_resp.status,
                error_text,
                endpoint="response",
                is_streaming=True,
            )
            return web.json_response(
                {"error": f"Upstream API error: {upstream_resp.status} {error_text}"},
                status=upstream_resp.status,
                content_type="application/json",
            )

        response_headers.update(
            {
                k: v
                for k, v in upstream_resp.headers.items()
                if k.lower()
                not in (
                    "content-type",
                    "content-encoding",
                    "transfer-encoding",
                    "content-length",  # in case of fake streaming
                )
            }
        )
        response = web.StreamResponse(
            status=upstream_resp.status,
            headers=response_headers,
        )
        response.enable_chunked_encoding()
        await response.prepare(request)

        # =======================================
        # Start event flow with ResponseCreatedEvent
        sequence_number = 0
        id = str(uuid.uuid4().hex)  # Generate a unique ID for the response

        onset_response = Response(
            id=f"resp_{id}",
            created_at=created_timestamp,
            model=data["model"],
            output=[],
            status="in_progress",
        )
        created_event = ResponseCreatedEvent(
            response=onset_response,
            sequence_number=sequence_number,
        )
        await send_off_sse(response, created_event.model_dump())

        # =======================================
        # ResponseInProgressEvent, start streaming the response
        sequence_number += 1
        in_progress_event = ResponseInProgressEvent(
            response=onset_response,
            sequence_number=sequence_number,
        )
        await send_off_sse(response, in_progress_event.model_dump())

        # =======================================
        # Collect upstream response for pseudo_stream so we can inspect tool calls
        # before deciding which output items to emit.
        cumulated_response = ""
        response_tool_calls = None
        if pseudo_stream:
            response_data = await upstream_resp.json()
            raw_response = response_data.get("response", "")

            # Run ToolInterceptor to extract tool calls
            if data.get("tools"):
                cs = ToolInterceptor()
                model_family = determine_model_family(data.get("model", ""))
                tool_calls_raw, cleaned_text = cs.process(
                    raw_response, model_family, request_data=data
                )
                cumulated_response = cleaned_text or ""
                if tool_calls_raw:
                    response_tool_calls = tool_calls_to_openai(
                        tool_calls_raw, api_format="response"
                    )
            else:
                cumulated_response = raw_response if isinstance(raw_response, str) else str(raw_response or "")
        else:
            # Real streaming — no tool interception (no pseudo_stream override)
            cumulated_response = ""

        # Log upstream response
        log_upstream_response(
            cumulated_response,
            endpoint="response",
            is_streaming=not pseudo_stream,
        )

        # =======================================
        output_index = 0
        output_msg = None

        if pseudo_stream:
            has_text = bool(cumulated_response.strip())
        else:
            # For real streaming we must open the output item before consuming
            # the upstream body because content is discovered incrementally.
            has_text = True

        if has_text:
            # ResponseOutputItemAddedEvent, add the text message item
            sequence_number += 1
            output_msg = ResponseOutputMessage(
                id=f"msg_{id}",
                content=[],
                status="in_progress",
            )
            output_item_event = ResponseOutputItemAddedEvent(
                item=output_msg,
                output_index=output_index,
                sequence_number=sequence_number,
            )
            await send_off_sse(response, output_item_event.model_dump())

            # ResponseContentPartAddedEvent
            sequence_number += 1
            content_part = ResponseContentPartAddedEvent(
                content_index=0,
                item_id=output_msg.id,
                output_index=output_index,
                part=ResponseOutputText(text=""),
                sequence_number=sequence_number,
            )
            await send_off_sse(response, content_part.model_dump())

            # Stream text chunks
            if pseudo_stream:
                chunk_size = 20
                for i in range(0, len(cumulated_response), chunk_size):
                    sequence_number += 1
                    chunk_text = cumulated_response[i : i + chunk_size]
                    text_delta = transform_streaming_response(
                        json.dumps({"response": chunk_text}),
                        content_index=0,
                        output_index=output_index,
                        sequence_number=sequence_number,
                        id=output_msg.id,
                    )
                    await send_off_sse(response, text_delta)
                    await asyncio.sleep(0.02)
            else:
                # Real streaming path
                decoder = StreamDecoder()
                async for chunk in upstream_resp.content.iter_any():
                    chunk_text, _ = decoder.decode(chunk)
                    if not chunk_text:
                        continue
                    sequence_number += 1
                    cumulated_response += chunk_text
                    text_delta = transform_streaming_response(
                        json.dumps({"response": chunk_text}),
                        content_index=0,
                        output_index=output_index,
                        sequence_number=sequence_number,
                        id=output_msg.id,
                    )
                    await send_off_sse(response, text_delta)
                remaining = decoder.flush()
                if remaining:
                    sequence_number += 1
                    cumulated_response += remaining
                    text_delta = transform_streaming_response(
                        json.dumps({"response": remaining}),
                        content_index=0,
                        output_index=output_index,
                        sequence_number=sequence_number,
                        id=output_msg.id,
                    )
                    await send_off_sse(response, text_delta)

            # ResponseTextDoneEvent
            sequence_number += 1
            text_done = ResponseTextDoneEvent(
                content_index=0,
                item_id=output_msg.id,
                output_index=output_index,
                sequence_number=sequence_number,
                text=cumulated_response,
            )
            await send_off_sse(response, text_done.model_dump())

            # ResponseContentPartDoneEvent
            sequence_number += 1
            output_text = ResponseOutputText(text=cumulated_response)
            content_part_done = ResponseContentPartDoneEvent(
                content_index=0,
                item_id=output_msg.id,
                output_index=output_index,
                part=output_text,
                sequence_number=sequence_number,
            )
            await send_off_sse(response, content_part_done.model_dump())

            # ResponseOutputItemDoneEvent
            sequence_number += 1
            output_msg.content = [output_text]
            output_msg.status = "completed"
            output_item_done = ResponseOutputItemDoneEvent(
                item=output_msg,
                output_index=output_index,
                sequence_number=sequence_number,
            )
            await send_off_sse(response, output_item_done.model_dump())

            if output_msg.content:
                onset_response.output.append(output_msg)
                output_index += 1

        # =======================================
        # Emit function_call output items with proper argument streaming events
        if response_tool_calls:
            for tc in response_tool_calls:
                # output_item.added (with in_progress status)
                sequence_number += 1
                tc.status = "in_progress"
                tc_added = ResponseOutputItemAddedEvent(
                    item=tc,
                    output_index=output_index,
                    sequence_number=sequence_number,
                )
                await send_off_sse(response, tc_added.model_dump())

                # function_call_arguments.delta (full arguments in one delta)
                sequence_number += 1
                args_delta = ResponseFunctionCallArgumentsDeltaEvent(
                    delta=tc.arguments,
                    item_id=tc.id,
                    output_index=output_index,
                    sequence_number=sequence_number,
                )
                await send_off_sse(response, args_delta.model_dump())

                # function_call_arguments.done
                sequence_number += 1
                args_done = ResponseFunctionCallArgumentsDoneEvent(
                    arguments=tc.arguments,
                    item_id=tc.id,
                    output_index=output_index,
                    sequence_number=sequence_number,
                )
                await send_off_sse(response, args_done.model_dump())

                # output_item.done (with completed status)
                sequence_number += 1
                tc.status = "completed"
                tc_done = ResponseOutputItemDoneEvent(
                    item=tc,
                    output_index=output_index,
                    sequence_number=sequence_number,
                )
                await send_off_sse(response, tc_done.model_dump())

                onset_response.output.append(tc)
                output_index += 1

        # =======================================
        # ResponseCompletedEvent, signal the end of the response
        sequence_number += 1
        onset_response.status = "completed"
        # Convert Pydantic tool call objects to dicts for token counting
        serializable_tcs = (
            [tc.model_dump() for tc in response_tool_calls]
            if response_tool_calls
            else None
        )
        output_tokens = await calculate_completion_tokens_async(
            cumulated_response,
            serializable_tcs,
            data["model"],
            api_format="response",
        )
        onset_response.usage = create_usage(
            prompt_tokens, output_tokens, api_type="response"
        )
        completed_event = ResponseCompletedEvent(
            response=onset_response,
            sequence_number=sequence_number,
        )
        await send_off_sse(response, completed_event.model_dump())

        # =======================================
        # Ensure response is properly closed

        await response.write_eof()

        return response


async def proxy_request(
    request: web.Request,
) -> Union[web.Response, web.StreamResponse]:
    """Proxies the client's request to an upstream API, handling response streaming and conversion.

    Args:
        request: The client's web request object.
        convert_to_openai: If True, translates the response to an OpenAI-compatible format.

    Returns:
        A web.Response or web.StreamResponse with the final response from the upstream API.
    """
    config: ArgoConfig = request.app["config"]
    model_registry: ModelRegistry = request.app["model_registry"]

    try:
        # Retrieve the incoming JSON data from request if input_data is not provided

        data = await request.json()
        stream = data.get("stream", False)
        # Force pseudo_stream when tools are present so we can
        # collect the full response and run ToolInterceptor on it
        pseudo_stream_override = "tools" in data

        if not data:
            raise ValueError("Invalid input. Expected JSON data.")

        # Log original request
        log_original_request(data, verbose=config.verbose)

        # Prepare the request data (includes message scrutinization and normalization)
        data = prepare_request_data(data, config, model_registry)

        # Apply username passthrough if enabled
        apply_username_passthrough(data, request, config.user)

        # Determine actual streaming mode for upstream request
        use_pseudo_stream = config.pseudo_stream or pseudo_stream_override
        if stream and use_pseudo_stream:
            # When using pseudo_stream, upstream request is non-streaming
            data["stream"] = False

        # Apply Claude max_tokens limit for non-streaming requests
        is_non_streaming_upstream = not stream or use_pseudo_stream
        data = apply_claude_max_tokens_limit(
            data, is_non_streaming=is_non_streaming_upstream
        )

        # Log converted request (now reflects actual upstream request mode)
        log_converted_request(data, verbose=config.verbose)

        # Use the shared HTTP session from app context for connection pooling
        session = request.app["http_session"]

        if stream:
            return await send_streaming_request(
                session,
                config,
                data,
                request,
                pseudo_stream=use_pseudo_stream,
            )
        else:
            return await send_non_streaming_request(
                session,
                config,
                data,
                convert_to_openai=True,
                openai_compat_fn=transform_non_streaming_response_async,
            )

    except ValueError as err:
        return web.json_response(
            {"error": str(err)},
            status=HTTPStatus.BAD_REQUEST,
            content_type="application/json",
        )
    except aiohttp.ClientError as err:
        error_message = f"HTTP error occurred: {err}"
        return web.json_response(
            {"error": error_message},
            status=HTTPStatus.SERVICE_UNAVAILABLE,
            content_type="application/json",
        )
    except Exception as err:
        error_message = f"An unexpected error occurred: {err}"
        return web.json_response(
            {"error": error_message},
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
            content_type="application/json",
        )
