#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import environ as env
from typing import Any, Optional, List, Dict

from openai import OpenAI

from llmdk.providers.interface import LlmInterface


class OpenAiClient(LlmInterface):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, **kwargs)

        if not api_key:
            api_key = env.get('OPENAI_API_KEY') or 'EMPTY'

        if not base_url:
            base_url = env.get('OPENAI_API_URL') or None

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> str:
        payload = self._generate_kwargs.copy()
        payload.update(kwargs)
        payload['model'] = self._model_name

        if messages is not None:
            payload['messages'] = messages
        else:
            payload['messages'] = []
            if system_prompt:
                payload['messages'].append({
                    'role': 'system',
                    'content': system_prompt,
                })
            payload['messages'].append({
                'role': 'user',
                'content': prompt,
            })

        completion = self._client.chat.completions.create(**payload)
        message = completion.choices[0].message.content
        return message
