#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import environ as env
from typing import Any, Optional, List, Dict

from ollama import Client

from llmdk.providers.interface import LlmInterface


class OllamaClient(LlmInterface):
    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        headers: Optional[dict] = None,
        options: Optional[dict] = None,
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, **kwargs)

        if not base_url:
            base_url = env.get('OLLAMA_API_URL') or 'http://localhost:11434'

        self._client = Client(
            host=base_url,
            headers=headers or {},
        )

        self._options = options or {}

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        options: Optional[dict] = None,
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

        merged_options = self._options.copy()
        if options:
            merged_options.update(options)
        payload['options'] = merged_options

        response = self._client.chat(**payload)
        message = response['message']['content']
        return message
