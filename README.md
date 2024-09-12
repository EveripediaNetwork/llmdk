<div align="center">
  <img src="./misc/llmdk.svg" alt="Logo" height="70" />
  <p><strong>Streamline LLM API Interactions</strong></p>
</div>
<br/>

# Installation

```bash
pip install .
```

# Usage

## Anthropic
```python
from llmdk import Llmdk, Providers

client = Llmdk(Providers.ANTHROPIC, 'claude-3-5-sonnet-20240620', api_key='***')
output = client.generate('Who are you?')
```

## Groq
```python
from llmdk import Llmdk, Providers

client = Llmdk(Providers.GROQ, 'llama-3.1-70b-versatile', api_key='***')
output = client.generate('Who are you?')
```

## HuggingFace
```python
from llmdk import Llmdk, Providers

client = Llmdk(Providers.HUGGINGFACE, 'meta-llama/Meta-Llama-3.1-70B-Instruct', api_key='***')
output = client.generate('Who are you?')
```

## Ollama
```python
from llmdk import Llmdk, Providers

client = Llmdk(Providers.OLLAMA, 'llama3.1:8b', base_url='http://...')
output = client.generate('Who are you?')
```

## OpenAI
```python
from llmdk import Llmdk, Providers

client = Llmdk(Providers.OPENAI, 'gpt-4o-2024-08-06', api_key='***')
output = client.generate('Who are you?')
```

## vLLM
```python
from llmdk import Llmdk, Providers

client = Llmdk(Providers.VLLM, base_url='http://...')
output = client.generate('Who are you?')
```
