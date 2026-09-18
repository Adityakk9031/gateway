from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from aiohttp import web

from speko_gateway.probe import ConversationProbe
from speko_gateway.relay import RelayEvaluationClient


class _FakeGatewayClient:
    def __init__(self) -> None:
        self.batches: list[list[dict[str, Any]]] = []

    async def post_turn_events(self, events: list[dict[str, Any]]) -> None:
        self.batches.append(list(events))

    async def aclose(self) -> None:
        pass


class _FakeEmitter:
    def __init__(self) -> None:
        self.handlers: dict[str, list[Any]] = {}

    def on(self, event: str, callback: Any) -> None:
        self.handlers.setdefault(event, []).append(callback)

    def off(self, event: str, callback: Any) -> None:
        if callback in self.handlers.get(event, []):
            self.handlers[event].remove(callback)

    def emit(self, event: str, payload: Any) -> None:
        for callback in list(self.handlers.get(event, [])):
            callback(payload)


class _FakeAgentSession(_FakeEmitter):
    def __init__(self) -> None:
        super().__init__()
        self.output = SimpleNamespace(audio=_FakeEmitter())
        self.user_state = "listening"


async def _server(handler):
    app = web.Application()
    app.router.add_post("/v1/evaluations", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


async def test_evaluation_client_sends_explicit_route_and_correlation():
    seen = {}

    async def handler(request):
        seen["body"] = await request.json()
        seen["idempotency"] = request.headers["Idempotency-Key"]
        seen["session"] = request.headers["Speko-Client-Session-ID"]
        return web.json_response(
            {
                "model": "jev-1.13.0",
                "answers": {"handoff": {"type": "noul", "noul": 0.97}},
                "usage": {"input_tokens": 100, "output_tokens": 4},
            },
            headers={"Speko-Request-ID": "req_eval_1"},
        )

    runner, base_url = await _server(handler)
    client = RelayEvaluationClient(
        api_key="test", base_url=base_url, session_id="conversation-1"
    )
    try:
        result = await client.evaluate(
            state={"transcript": "human please"},
            questions={
                "handoff": {
                    "type": "noul",
                    "instructions": "Was a human requested?",
                }
            },
            idempotency_key="turn-1",
        )
    finally:
        await client.aclose()
        await runner.cleanup()

    assert result["answers"]["handoff"]["noul"] == 0.97
    assert seen["body"]["routing"] == {
        "mode": "explicit",
        "provider": "typesafe",
        "model": "jev-1.13.0",
    }
    assert seen["idempotency"] == "turn-1"
    assert seen["session"] == "conversation-1"


async def test_evaluation_client_propagates_error_and_cancellation():
    started = asyncio.Event()

    async def handler(request):
        started.set()
        await asyncio.sleep(0.2)
        return web.json_response({})

    runner, base_url = await _server(handler)
    client = RelayEvaluationClient(api_key="test", base_url=base_url)
    task = asyncio.create_task(
        client.evaluate(
            state="hello",
            questions={
                "intent": {
                    "type": "choice",
                    "instructions": "intent",
                    "criteria": {"other": None, "status": None},
                }
            },
            idempotency_key="turn-cancel",
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await client.aclose()
    await runner.cleanup()


async def test_evaluation_client_requires_explicit_idempotency_key():
    client = RelayEvaluationClient(api_key="test", base_url="http://127.0.0.1:1")
    try:
        with pytest.raises(ValueError, match="idempotency_key"):
            await client.evaluate(state="x", questions={}, idempotency_key="")
    finally:
        await client.aclose()


async def test_delayed_evaluation_is_not_attached_to_a_later_turn():
    request_started = asyncio.Event()
    release_response = asyncio.Event()

    async def handler(request):
        request_started.set()
        await release_response.wait()
        return web.json_response(
            {
                "model": "jev-1.13.0",
                "answers": {"handoff": {"type": "noul", "noul": 0.1}},
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
            headers={"Speko-Request-ID": "req_delayed"},
        )

    runner, base_url = await _server(handler)
    session = _FakeAgentSession()
    probe_client = _FakeGatewayClient()
    probe = ConversationProbe(session, client=probe_client)  # type: ignore[arg-type]
    probe.start()
    session.user_state = "speaking"
    session.emit(
        "user_state_changed",
        SimpleNamespace(old_state="listening", new_state="speaking"),
    )
    client = RelayEvaluationClient(api_key="test", base_url=base_url)
    task = asyncio.create_task(
        client.evaluate(
            state={"transcript": "first turn"},
            questions={
                "handoff": {
                    "type": "noul",
                    "instructions": "Was a human requested?",
                }
            },
            idempotency_key="turn-delayed",
        )
    )
    try:
        await request_started.wait()
        probe._complete_turn()
        probe._begin_turn("user")
        release_response.set()
        await task
    finally:
        await client.aclose()
        await probe.aclose()
        await runner.cleanup()

    events = [event for batch in probe_client.batches for event in batch]
    assert not any(
        event["type"] == "leg.attached"
        and event["data"].get("request_id") == "req_delayed"
        for event in events
    )
