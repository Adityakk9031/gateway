from __future__ import annotations

import asyncio
from typing import Any

from examples.jev_voice_router import VoiceTurnRouter


def _response(
    *, choice: str = "order_status", probability: float = 0.95,
    confidence: float = 0.95, handoff: float = 0.01,
) -> dict[str, Any]:
    remainder = 1 - probability
    return {
        "model": "jev-1.13.0",
        "answers": {
            "intent": {
                "type": "choice",
                "choice": choice,
                "probabilities": {
                    "order_status": probability if choice == "order_status" else remainder / 2,
                    "store_hours": probability if choice == "store_hours" else remainder / 2,
                    "other": probability if choice == "other" else remainder / 2,
                },
                "confidence": confidence,
            },
            "handoff_requested": {"type": "noul", "noul": handoff},
            "frustration": {
                "type": "score", "score": 0.0,
                "probabilities": {"0": 1.0, "1": 0.0}, "confidence": 1.0,
                "legend": {"0": "calm", "1": "frustrated"},
            },
        },
        "usage": {"input_tokens": 20, "output_tokens": 2},
    }


class Evaluator:
    def __init__(self, responses: list[dict[str, Any]], delay: float = 0) -> None:
        self.responses = responses
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.calls = 0

    async def evaluate(self, **_: Any) -> dict[str, Any]:
        index = self.calls
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            return self.responses[min(index, len(self.responses) - 1)]
        finally:
            self.active -= 1


def _router(evaluator: Evaluator, spoken: list[str], *, deadline: float = 0.5):
    async def normal(transcript: str, _: dict[str, Any]) -> str:
        return f"llm:{transcript}"

    async def handoff(_: str, __: dict[str, Any]) -> str:
        return "human"

    async def status(_: str, __: dict[str, Any]) -> str:
        return "status"

    async def speak(text: str) -> None:
        spoken.append(text)

    return VoiceTurnRouter(
        evaluator, normal_llm=normal, human_handoff=handoff, speak=speak,
        read_only_handlers={"order_status": status}, deadline_seconds=deadline,
    )


async def test_voice_router_dispatches_only_certain_allowlisted_choice():
    spoken: list[str] = []
    router = _router(Evaluator([_response()]), spoken)
    await router.on_final_transcript("where is my order", {})
    assert spoken == ["status"]


async def test_voice_router_uncertainty_other_and_timeout_use_llm():
    for response, delay, deadline in [
        (_response(probability=0.89), 0, 0.5),
        (_response(confidence=0.89), 0, 0.5),
        (_response(choice="other", probability=0.95), 0, 0.5),
        (_response(), 0.05, 0.001),
    ]:
        spoken: list[str] = []
        router = _router(Evaluator([response], delay), spoken, deadline=deadline)
        await router.on_final_transcript("question", {})
        assert spoken == ["llm:question"]


async def test_voice_router_handoff_threshold_is_independent():
    spoken: list[str] = []
    router = _router(Evaluator([_response(probability=0.99, confidence=0.99, handoff=0.9)]), spoken)
    await router.on_final_transcript("person please", {})
    assert spoken == ["human"]


async def test_voice_router_discards_superseded_turn_and_bounds_work():
    spoken: list[str] = []
    evaluator = Evaluator([_response(), _response(choice="other")], delay=0.05)
    router = _router(evaluator, spoken)
    first = asyncio.create_task(router.on_final_transcript("first", {}))
    await asyncio.sleep(0.005)
    second = asyncio.create_task(router.on_final_transcript("second", {}))
    await asyncio.gather(first, second)
    assert spoken == ["llm:second"]
    assert evaluator.max_active == 1
