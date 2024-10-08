<div align="center">
  <img src="./misc/llmdk.svg" alt="Logo" height="70" />
  <p><strong>Streamline LLM Interactions in Python</strong></p>
</div>
<br/>

<p align="center">
    <a href="https://pypi.python.org/pypi/llmdk/"><img alt="PyPi" src="https://img.shields.io/pypi/v/llmdk.svg?style=flat-square"></a>
    <a href="https://github.com/EveripediaNetwork/llmdk/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/github/license/EveripediaNetwork/llmdk.svg?style=flat-square"></a>
</p>

# Installation

```bash
pip install llmdk
```

# Basic Usage

## Client

```python
from llmdk import Llmdk, Providers

# You can also set OPENAI_API_KEY
client = Llmdk(
    provider=Providers.OPENAI,
    model_name='gpt-4o-mini',
    # api_key='***',
)
```

## Generate

### Prompt

```python
output = client.generate(
    'Who are you?',
    # system='Write in Portuguese.',
)
```

### List of messages

```python
output = client.generate(
    messages=[
        # {'role': 'system', 'content': 'Write in Portuguese.'},
        {'role': 'user', 'content': 'Who are you?'},
    ],
)
```

## Stream

### Prompt

```python
for chunk in client.stream(
    'Who are you?',
    # system='Write in Portuguese.',
):
    print(chunk, end='', flush=True)
```

### List of messages

```python
for chunk in client.stream([
    # {'role': 'system', 'content': 'Write in Portuguese.'},
    {'role': 'user', 'content': 'Who are you?'},
]):
    print(chunk, end='', flush=True)
```

# Supported Providers

## Anthropic

```python
from llmdk import Llmdk, Providers

# You can also set ANTHROPIC_API_KEY
client = Llmdk(
    provider=Providers.ANTHROPIC,
    model_name='claude-3-5-sonnet-20240620',
    # api_key='***',
)
```

## Groq

```python
from llmdk import Llmdk, Providers

# You can also set GROQ_API_KEY
client = Llmdk(
    provider=Providers.GROQ,
    model_name='llama-3.1-70b-versatile',
    # api_key='***',
)
```

## HuggingFace

```python
from llmdk import Llmdk, Providers

# You can also set HF_TOKEN
client = Llmdk(
    provider=Providers.HUGGINGFACE,
    model_name='meta-llama/Meta-Llama-3.1-70B-Instruct',
    # api_key='***',
)
```

## Ollama

```python
from llmdk import Llmdk, Providers

client = Llmdk(
    provider=Providers.OLLAMA,
    model_name='llama3.2:1b',
    # base_url='http://localhost:11434',
)
```

## OpenAI

```python
from llmdk import Llmdk, Providers

# You can also set OPENAI_API_KEY
client = Llmdk(
    provider=Providers.OPENAI,
    model_name='gpt-4o-mini',
    # api_key='***',
)
```
