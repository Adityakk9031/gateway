"""Framework-independent voice routing example using one batched Jev evaluation.

Run without TypeSafe credentials:

    python examples/jev_voice_router.py --fake

The voice framework remains responsible for VAD and turn detection. Call
``on_final_transcript`` only after a turn is finalized.
"""

from __future__ import annotations

import argparse
import asyncio
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from aiohttp import ClientError, web

from speko_gateway.relay import (
    EvaluationQuestion,
    EvaluationResponse,
    RelayError,
    RelayEvaluationClient,
)

CHOICE_THRESHOLD = 0.9
CHOICE_CONFIDENCE_THRESHOLD = 0.9
HANDOFF_THRESHOLD = 0.9
# These are example defaults. Validate and calibrate them on your own data.

QUESTIONS: dict[str, EvaluationQuestion] = {
    "intent": {
        "type": "choice",
        "instructions": "Classify the customer's immediate intent.",
        "criteria": {
            "order_status": "Read the current status of an existing order.",
            "store_hours": "Ask when the store is open.",
            "other": "Anything else or anything uncertain.",
        },
    },
    "handoff_requested": {
        "type": "noul",
        "instructions": "Is the customer explicitly requesting a human?",
    },
    "frustration": {
        "type": "score",
        "instructions": "Rate customer frustration using the ordered rubric.",
        "criteria": [
            "calm or neutral",
            "mildly frustrated",
            "clearly frustrated",
            "very frustrated or angry",
        ],
    },
}


class Evaluator(Protocol):
    async def evaluate(
        self,
        *,
        state: Any,
        questions: dict[str, EvaluationQuestion],
        idempotency_key: str,
        provider: str = "typesafe",
        model: str = "jev-1.13.0",
    ) -> EvaluationResponse: ...


AsyncTextHandler = Callable[[str, dict[str, Any]], Awaitable[str]]


class VoiceTurnRouter:
    """Routes finalized turns while bounding evaluation work per conversation."""

    def __init__(
        self,
        evaluator: Evaluator,
        *,
        normal_llm: AsyncTextHandler,
        human_handoff: AsyncTextHandler,
        speak: Callable[[str], Awaitable[None]],
        read_only_handlers: dict[str, AsyncTextHandler],
        deadline_seconds: float = 0.5,
    ) -> None:
        self._evaluator = evaluator
        self._normal_llm = normal_llm
        self._human_handoff = human_handoff
        self._speak = speak
        self._handlers = dict(read_only_handlers)
        self._deadline = deadline_seconds
        self._turn = 0
        self._in_flight: asyncio.Task[EvaluationResponse] | None = None
        self._evaluation_lock = asyncio.Lock()

    async def on_final_transcript(
        self, transcript: str, relevant_state: dict[str, Any]
    ) -> None:
        """Evaluate the current turn and speak only if it is still current."""

        async with self._evaluation_lock:
            self._turn += 1
            turn = self._turn
            previous = self._in_flight
            if previous is not None and not previous.done():
                previous.cancel()
                await asyncio.gather(previous, return_exceptions=True)
            task = asyncio.create_task(
                self._evaluator.evaluate(
                    state={"transcript": transcript, "application": relevant_state},
                    questions=QUESTIONS,
                    idempotency_key=f"voice-turn-{turn}-{uuid.uuid4()}",
                )
            )
            self._in_flight = task
        try:
            evaluation = await asyncio.wait_for(task, timeout=self._deadline)
        except asyncio.CancelledError:
            if turn != self._turn:
                return
            raise
        except (asyncio.TimeoutError, ClientError, RelayError):
            if turn != self._turn:
                return
            reply = await self._normal_llm(transcript, relevant_state)
            if turn == self._turn:
                await self._speak(reply)
            return
        if turn != self._turn:
            return

        answers = evaluation["answers"]
        handoff = answers.get("handoff_requested", {})
        if float(handoff.get("noul", 0.0)) >= HANDOFF_THRESHOLD:
            reply = await self._human_handoff(transcript, relevant_state)
            if turn == self._turn:
                await self._speak(reply)
            return

        intent = answers.get("intent", {})
        choice = str(intent.get("choice", "other"))
        probabilities = intent.get("probabilities", {})
        winner_probability = float(probabilities.get(choice, 0.0))
        confidence = float(intent.get("confidence", 0.0))
        handler = self._handlers.get(choice)
        if (
            choice != "other"
            and handler is not None
            and winner_probability >= CHOICE_THRESHOLD
            and confidence >= CHOICE_CONFIDENCE_THRESHOLD
        ):
            # Only an allowlisted read-only handler can be dispatched here.
            # Calculations and authorization for side effects stay in code.
            reply = await handler(transcript, relevant_state)
            if turn == self._turn:
                await self._speak(reply)
            return
        reply = await self._normal_llm(transcript, relevant_state)
        if turn == self._turn:
            await self._speak(reply)


async def fake_evaluation(request: web.Request) -> web.Response:
    body = await request.json()
    transcript = str(body.get("state", {}).get("transcript", "")).lower()
    handoff = 0.97 if any(word in transcript for word in ("human", "person")) else 0.02
    if "order" in transcript:
        choice, probabilities, confidence = (
            "order_status",
            {"order_status": 0.95, "store_hours": 0.01, "other": 0.04},
            0.94,
        )
    elif "hours" in transcript or "open" in transcript:
        choice, probabilities, confidence = (
            "store_hours",
            {"order_status": 0.01, "store_hours": 0.96, "other": 0.03},
            0.95,
        )
    else:
        choice, probabilities, confidence = (
            "other",
            {"order_status": 0.1, "store_hours": 0.1, "other": 0.8},
            0.55,
        )
    return web.json_response(
        {
            "model": "jev-1.13.0",
            "answers": {
                "intent": {
                    "type": "choice",
                    "choice": choice,
                    "probabilities": probabilities,
                    "confidence": confidence,
                },
                "handoff_requested": {"type": "noul", "noul": handoff},
                "frustration": {
                    "type": "score",
                    "score": 0,
                    "probabilities": {"0": 0.95, "1": 0.03, "2": 0.01, "3": 0.01},
                    "confidence": 0.93,
                    "legend": {
                        "0": "calm or neutral",
                        "1": "mildly frustrated",
                        "2": "clearly frustrated",
                        "3": "very frustrated or angry",
                    },
                },
            },
            "usage": {"input_tokens": 180, "output_tokens": 12},
        },
        headers={"Speko-Request-ID": "req_fake_jev"},
    )


async def run_demo(use_fake: bool) -> None:
    runner: web.AppRunner | None = None
    if use_fake:
        app = web.Application()
        app.router.add_post("/v1/evaluations", fake_evaluation)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        client = RelayEvaluationClient(api_key="fake", base_url=f"http://127.0.0.1:{port}")
    else:
        client = RelayEvaluationClient.from_env(session_id="voice-example")

    async def normal_llm(transcript: str, _: dict[str, Any]) -> str:
        return f"Normal LLM flow: {transcript}"

    async def handoff(_: str, __: dict[str, Any]) -> str:
        return "I’ll connect you with a person."

    async def order_status(_: str, state: dict[str, Any]) -> str:
        return f"Order {state.get('order_id', 'unknown')} is being prepared."

    async def store_hours(_: str, __: dict[str, Any]) -> str:
        return "The store is open from 9 AM to 6 PM."

    async def speak(text: str) -> None:
        print(f"Assistant: {text}")

    router = VoiceTurnRouter(
        client,
        normal_llm=normal_llm,
        human_handoff=handoff,
        speak=speak,
        read_only_handlers={
            "order_status": order_status,
            "store_hours": store_hours,
        },
    )
    try:
        while transcript := await asyncio.to_thread(input, "Final transcript (blank exits): "):
            await router.on_final_transcript(transcript, {"order_id": "A-1042"})
    finally:
        await client.aclose()
        if runner is not None:
            await runner.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", action="store_true", help="run a local fake evaluation server")
    arguments = parser.parse_args()
    asyncio.run(run_demo(arguments.fake))
