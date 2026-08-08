from __future__ import annotations

import json
import os
from contextlib import ExitStack
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.servers import edge_first_proxy as proxy


class _DummyExpertDataPlane:
    def __init__(self) -> None:
        self.advance_calls = 0
        self.begin_calls = 0
        self.complete_calls = 0

    def begin_request(self, **kwargs):
        self.begin_calls += 1
        return object()

    def advance_window(self, session) -> None:
        self.advance_calls += 1

    def complete_request(self, session, *, success, response_text) -> None:
        self.complete_calls += 1

    def runtime_snapshot(self) -> dict:
        return {}


class _FakeContent:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_any(self):
        for chunk in self._chunks:
            yield chunk


class _FakeResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self.headers = {"content-type": "text/event-stream"}
        self.content = _FakeContent(chunks)

    def release(self) -> None:
        return None


class _FakeCloudSession:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def post(self, *args, **kwargs):
        return _FakeResponse(self._chunks)


class _WhitespaceTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return [token for token in text.split(" ") if token]


def _sse_chunk(content: str | None = None, finish_reason: str | None = None) -> bytes:
    payload = {
        "id": "cloud_1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "edge-first",
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    if content is not None:
        payload["choices"][0]["delta"]["content"] = content
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


class EdgeFirstFrontierRegressionTest(unittest.TestCase):
    def test_edge_first_frontier_advance_matches_emitted_content(self) -> None:
        cases = [
            (
                "spec_hit",
                True,
                True,
                "one ",
                [
                    _sse_chunk("one "),
                    _sse_chunk("two three four"),
                    b"data: [DONE]\n\n",
                ],
                "one two three four",
            ),
            (
                "spec_miss",
                True,
                True,
                "zero ",
                [
                    _sse_chunk("one "),
                    _sse_chunk("two three four"),
                    b"data: [DONE]\n\n",
                ],
                "zero one two three four",
            ),
            (
                "no_spec",
                False,
                False,
                None,
                [
                    _sse_chunk("one two "),
                    _sse_chunk("three four"),
                    b"data: [DONE]\n\n",
                ],
                "one two three four",
            ),
        ]
        advance_interval = 2

        for case_name, draft_enabled, should_speculate, predicted_text, cloud_chunks, expected_text in cases:
            with self.subTest(case=case_name):
                dummy_plane = _DummyExpertDataPlane()

                async def _fake_get_cloud_session():
                    return _FakeCloudSession(cloud_chunks)

                with ExitStack() as stack:
                    stack.enter_context(
                        patch.dict(
                            os.environ,
                            {
                                "EDGE_FIRST_ENABLED": "1",
                                "EDGE_EXPERT_ADVANCE_TOKEN_INTERVAL": str(advance_interval),
                                "CLOUD_URL": "http://fake-cloud.local",
                            },
                            clear=False,
                        )
                    )
                    stack.enter_context(patch.object(proxy, "_expert_data_plane", dummy_plane))
                    stack.enter_context(
                        patch.object(
                            proxy,
                            "_build_draft_policy_from_hermes",
                            lambda body, family_info, transport_route: (
                                {
                                    "draft_mode": "mtp" if draft_enabled else "off",
                                    "response_contract": "plain",
                                    "grammar_mode": "off",
                                    "enabled": draft_enabled,
                                    "disable_reason": "" if draft_enabled else "manual_test_disabled",
                                    "roi": 0.1,
                                    "json_success_rate": 1.0,
                                    "source": "test",
                                },
                                {},
                                {"mode": proxy.ROUTE_CLOUD_PD, "pivot_layer": 0, "reason": case_name},
                            ),
                        )
                    )
                    stack.enter_context(patch.object(proxy, "_should_speculate", lambda family_info: should_speculate))
                    stack.enter_context(
                        patch.object(proxy, "_should_speculate_for_temperature", lambda temperature, family_info: True)
                    )
                    stack.enter_context(
                        patch.object(proxy, "_edge_generate_first_token", lambda messages, max_tokens=1: predicted_text)
                    )
                    stack.enter_context(patch.object(proxy, "_maybe_schedule_warmup", lambda *args, **kwargs: None))
                    stack.enter_context(patch.object(proxy, "_load_edge_tokenizer", lambda: _WhitespaceTokenizer()))
                    stack.enter_context(patch.object(proxy, "_get_cloud_session", _fake_get_cloud_session))

                    client = TestClient(proxy.app)
                    emitted_chunks: list[str] = []
                    with client.stream(
                        "POST",
                        "/v1/chat/completions",
                        json={
                            "model": "edge-first-smoke",
                            "stream": True,
                            "temperature": 0.0,
                            "messages": [{"role": "user", "content": "say tokens"}],
                            "max_tokens": 16,
                        },
                    ) as resp:
                        self.assertEqual(resp.status_code, 200)
                        for line in resp.iter_lines():
                            if not line or not line.startswith("data: "):
                                continue
                            payload = line[6:]
                            if payload == "[DONE]":
                                continue
                            obj = json.loads(payload)
                            choices = obj.get("choices") or []
                            if not choices:
                                continue
                            content = str((choices[0].get("delta") or {}).get("content") or "")
                            if content:
                                emitted_chunks.append(content)

                emitted_text = "".join(emitted_chunks)
                emitted_tokens = len([token for token in emitted_text.split(" ") if token])
                expected_advances = emitted_tokens // advance_interval

                self.assertEqual(emitted_text, expected_text)
                self.assertEqual(dummy_plane.begin_calls, 1)
                self.assertEqual(dummy_plane.complete_calls, 1)
                self.assertEqual(dummy_plane.advance_calls, expected_advances)
