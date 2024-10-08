#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import environ as env
from typing import Any, Iterator, Optional, Dict
from anthropic import Anthropic
from llmdk.providers.interface import LlmInterface


class AnthropicClient(LlmInterface):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, **kwargs)

        if not api_key:
            api_key = env.get('ANTHROPIC_API_KEY')

        self._client = Anthropic(
            api_key=api_key,
            base_url=base_url,
        )

    def _prepare_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if 'max_tokens' not in payload:
            payload['max_tokens'] = 4096  # Required by Anthropic

        # Check for system message and move it to the system property
        messages = payload.get('messages', [])
        if messages and messages[0]['role'] == 'system':
            payload['system'] = messages[0]['content']
            payload['messages'] = messages[1:]

        return payload

    def _execute_request(self, payload: Dict[str, Any]) -> str:
        prepared_payload = self._prepare_payload(payload)
        completion = self._client.messages.create(**prepared_payload)
        return completion.content[0].text

    def _execute_stream_request(
        self,
        payload: Dict[str, Any],
    ) -> Iterator[str]:
        prepared_payload = self._prepare_payload(payload)

        with self._client.messages.stream(**prepared_payload) as stream:
            for message in stream:
                if message.type == 'content_block_delta':
                    yield message.delta.text
