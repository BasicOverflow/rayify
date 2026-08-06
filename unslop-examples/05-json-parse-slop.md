# Structured-output “JSON recovery” slop

## Slop tells

- Pretend freeform LLM text is almost JSON and “helpfully” recover it
- Regex that does not nest: `\{[^{}]*\}`
- Brace-counting extractors after guided_json already guarantees schema
- Prompt stuffing a full schema into the user message then re-parsing with try/except soup

---

## AI-slop: actor invents structured output via prompt + regex

```python
def generate_with_schema(self, prompt: str, pydantic_class: Type[BaseModel], **kwargs):
    """Generate structured output using Pydantic schema."""
    schema_str = json.dumps(pydantic_class.model_json_schema(), indent=2)
    enhanced_prompt = f"{prompt}\n\nRespond in valid JSON matching this schema:\n{schema_str}"

    result = self.generate(enhanced_prompt, **kwargs)
    text = result[0] if isinstance(result, list) and result else str(result)

    import re
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            return pydantic_class(**parsed)
        except:
            pass

    try:
        parsed = json.loads(text)
        return pydantic_class(**parsed)
    except:
        raise ValueError(f"Could not parse structured output from: {text}")
```

## Halfway unslop: use guided_json on generate, still keep a fat parser in inference

```python
def generate_with_schema(self, prompt: str, pydantic_class: Type[BaseModel], **kwargs):
    """Generate structured output using vLLM's native guided_json support."""
    json_schema = pydantic_class.model_json_schema()
    result = self.generate(prompt, guided_json=json_schema, **kwargs)
    text = result[0] if isinstance(result, list) and result else str(result)
    parsed = json.loads(text)
    return pydantic_class(**parsed)
```

Meanwhile inference still did “just in case”:

```python
def _parse_structured_output(text: str, pydantic_class: Type[BaseModel]):
    """Parse structured output from text. vLLM's guided_json ensures valid JSON."""
    import json

    text = text.strip()
    try:
        return pydantic_class(**json.loads(text))
    except json.JSONDecodeError:
        start = text.find('{')
        if start != -1:
            count = 0
            for i in range(start, len(text)):
                if text[i] == '{': count += 1
                elif text[i] == '}': count -= 1
                if count == 0:
                    try:
                        return pydantic_class(**json.loads(text[start:i+1]))
                    except json.JSONDecodeError:
                        break
        raise ValueError(f"Could not parse JSON from model output. Text: {text[:200]}")
```

> Docstring admits guided_json ensures valid JSON — then spends 20 lines not trusting that.

## Fully unslopified

```python
def _parse_structured_output(text: str, pydantic_class: Type[BaseModel]):
    import json
    return pydantic_class(**json.loads(text.strip()))
```

If it isn’t JSON, it throws. That’s the point.

---

## Parallel prior form (regex-only recovery)

```python
def _parse_structured_output(text: str, pydantic_class: Type[BaseModel]):
    """Parse structured output from text."""
    import json
    import re

    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            return pydantic_class(**parsed)
        except:
            pass

    try:
        parsed = json.loads(text)
        return pydantic_class(**parsed)
    except:
        raise ValueError(f"Could not parse structured output from: {text}")
```

Classic AI pattern: three different “parse maybe” strategies in one function, bare `except:`, and a broken nested-object regex.
