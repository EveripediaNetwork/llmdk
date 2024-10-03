#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import environ as env
from typing import Any, Optional, List, Dict

from huggingface_hub import InferenceClient

from llmdk.providers.interface import LlmInterface


class HuggingFaceClient(LlmInterface):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, **kwargs)

        if not api_key:
            api_key = env.get('HF_TOKEN')

        self._client = InferenceClient(
            model=model_name,
            token=api_key,
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

        completion = self._client.chat_completion(**payload)
        message = completion.choices[0].message.content
        return message
