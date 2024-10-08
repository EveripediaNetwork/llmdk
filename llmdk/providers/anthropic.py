#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import environ as env
from typing import Any, Optional, Dict
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

    def _execute_request(self, payload: Dict[str, Any]) -> str:
        payload['max_tokens'] = 4096  # Required
        completion = self._client.messages.create(**payload)
        return completion.content[0].text
