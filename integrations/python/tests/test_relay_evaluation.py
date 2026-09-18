from __future__ import annotations

import asyncio

import pytest
from aiohttp import web

from speko_gateway.relay import RelayEvaluationClient


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
