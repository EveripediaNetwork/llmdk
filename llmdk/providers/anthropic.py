#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import environ as env
from typing import Any, Optional

from anthropic import Anthropic

from llmdk.providers.interface import LlmInterface


class AnthropicModels:
    # Claude 3.5
    CLAUDE_3_5_SONNET_20240620 = 'claude-3-5-sonnet-20240620'
    CLAUDE_3_5_SONNET = CLAUDE_3_5_SONNET_20240620

    # Claude 3
    CLAUDE_3_OPUS_20240229 = 'claude-3-opus-20240229'
    CLAUDE_3_OPUS = CLAUDE_3_OPUS_20240229
    CLAUDE_3_SONNET_20240229 = 'claude-3-sonnet-20240229'
    CLAUDE_3_SONNET = CLAUDE_3_SONNET_20240229
    CLAUDE_3_HAIKU_20240307 = 'claude-3-haiku-20240307'
    CLAUDE_3_HAIKU = CLAUDE_3_HAIKU_20240307

    # Legacy
    CLAUDE_2_1 = 'claude-2.1'
    CLAUDE_2_0 = 'claude-2.0'
    CLAUDE_INSTANT_1_2 = 'claude-instant-1.2'


class AnthropicClient(LlmInterface):
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        if not model_name:
            model_name = env.get('ANTHROPIC_MODEL', AnthropicModels.CLAUDE_3_SONNET)
        super().__init__(model_name=model_name, **kwargs)

        if not api_key:
            api_key = env.get('ANTHROPIC_API_KEY')

        self._client = Anthropic(
            api_key=api_key,
            base_url=base_url,
        )

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        payload = {
            'model': self._model_name,
            'messages': [{
                "role": "user",
                "content": prompt,
            }],
        }

        if temperature is not None:
            payload['temperature'] = temperature

        # Required by Anthropic
        if max_tokens is None:
            max_tokens = 4096
        payload['max_tokens'] = max_tokens

        message = self._client.messages.create(**payload).content[0].text
        return message
