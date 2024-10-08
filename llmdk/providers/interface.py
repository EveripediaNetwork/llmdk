#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Optional, Iterator, Union


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
        prompt_or_messages: Union[str, List[Dict[str, str]]],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        payload = self._generate_payload(
            prompt_or_messages,
            system,
            **kwargs,
        )
        return self._execute_request(payload)

    def stream(
        self,
        prompt_or_messages: Union[str, List[Dict[str, str]]],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        payload = self._generate_payload(
            prompt_or_messages,
            system,
            **kwargs,
        )
        yield from self._execute_stream_request(payload)

    def _generate_payload(
        self,
        prompt_or_messages: Union[str, List[Dict[str, str]]],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        payload = self._generate_kwargs.copy()
        payload.update(kwargs)
        payload['model'] = self._model_name
        payload['messages'] = self._prepare_messages(
            prompt_or_messages,
            system,
        )
        return payload

    def _prepare_messages(
        self,
        prompt_or_messages: Union[str, List[Dict[str, str]]],
        system: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        if isinstance(prompt_or_messages, list):
            prepared_messages = prompt_or_messages.copy()
            if system:
                self._add_system(prepared_messages, system)
        else:
            prepared_messages = []
            if system:
                prepared_messages.append({
                    'role': 'system',
                    'content': system,
                })
            if prompt_or_messages:
                prepared_messages.append({
                    'role': 'user',
                    'content': prompt_or_messages,
                })
        return prepared_messages

    def _add_system(
        self,
        messages: List[Dict[str, str]],
        system: str,
    ) -> None:
        if messages and messages[0]['role'] == 'system':
            messages[0]['content'] = system
        else:
            messages.insert(0, {
                'role': 'system',
                'content': system,
            })

    def _execute_request(self, payload: Dict[str, Any]) -> str:
        raise NotImplementedError('Subclasses must implement this method')

    def _execute_stream_request(
        self,
        payload: Dict[str, Any],
    ) -> Iterator[str]:
        raise NotImplementedError(
            'Streaming not implemented for this provider',
        )
