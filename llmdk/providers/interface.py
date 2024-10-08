#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Optional


class LlmInterface:
    def __init__(
        self,
        model_name: Optional[str] = None,
        **kwargs: Any,
    ):
        self._model_name = model_name
        self._generate_kwargs = kwargs

    @property
    def model_name(self) -> Optional[str]:
        return self._model_name

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
            if system_prompt:
                self._add_system_prompt(payload['messages'], system_prompt)
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

        return self._execute_request(payload)

    def _add_system_prompt(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
    ) -> None:
        if messages and messages[0]['role'] == 'system':
            messages[0]['content'] = system_prompt
        else:
            messages.insert(0, {
                'role': 'system',
                'content': system_prompt,
            })

    def _execute_request(self, payload: Dict[str, Any]) -> str:
        raise NotImplementedError("Subclasses must implement this method")
