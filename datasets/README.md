# Datasets

## washing_machine_corpus

Filtered subset of WikiText-2 containing sentences with "washing machine", "washing", or "machine".

- **Source:** WikiText-2-raw-v1
- **Size:** ~350 training examples
- **Format:** HuggingFace Dataset (arrow)

### Loading

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/washing_machine_corpus")
```

### Samples
See `datasets/washing_machine_corpus/samples.json`
