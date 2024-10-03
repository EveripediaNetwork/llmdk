#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, Optional, List

import requests

from llmdk.providers.interface import LlmInterface


class VllmClient(LlmInterface):
    def __init__(
        self,
        base_url: str,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._base_url = base_url

    def _post(self, payload: Dict) -> Dict:
        response = requests.post(
            self._base_url,
            json={
                **payload,
                'stream': False,
            },
        )
        return response.json()

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> str:
        payload = self._generate_kwargs.copy()
        payload.update(kwargs)

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

        response = self._post(payload)
        message = response['choices'][0]['message']['content']
        return message
