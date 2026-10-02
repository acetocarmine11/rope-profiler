# Question/context sampling

The paper samples question queries and context keys. Region masks let the
monitor use that eligibility population, while keeping the caller's existing
model inputs. Masks must refer to the exact tokenized prompt including its chat
template. Keep special tokens, padding and punctuation-only overlaps excluded.

This runnable example uses character offsets without loading a tokenizer or
checkpoint:

```python
import torch
from rope_profiler import MonitorConfig, RoPEMonitor, token_region_masks
from transformers import LlamaConfig, LlamaForCausalLM

text = "Paris is in France. Where is Paris?"
# Synthetic tokenization of the exact text, with one leading special token.
offsets = [[[0, 0], [0, 5], [6, 8], [9, 11], [12, 18],
            [18, 19], [20, 25], [26, 28], [29, 34], [34, 35]]]
masks = token_region_masks(
    [text], offsets,
    context_spans=[(0, 19)], question_spans=[(20, 35)],
    special_tokens_mask=[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
)
ids = torch.arange(1, 11).unsqueeze(0)  # Synthetic IDs matching these offsets.
model = LlamaForCausalLM(LlamaConfig(
    vocab_size=32, hidden_size=16, intermediate_size=32,
    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
)).eval()
monitor = RoPEMonitor(model, MonitorConfig(sampling_policy="regions"))
with torch.inference_mode(), monitor.evaluation(step=0):
    with monitor.batch(ids, **masks):
        model(input_ids=ids)
monitor.save_report("outputs/regions_report.json")
```

For real inputs use the evaluated model's tokenizer, request offset and special
token masks, and supply task-preparation character spans. `token_region_masks`
checks those spans and returns CPU Boolean masks. It accepts nonempty token
overlaps only when a letter/digit exists inside that region. The causal sampler
also requires every chosen key to precede every chosen query.

Trainer integration uses the same masks through a callback:

```python
from rope_profiler import MonitorConfig, attach_rope_monitor

# This closure may look up precomputed masks by effective input fingerprint.
# Build/stack the masks in the actual collated batch order.
def batch_metadata(inputs):
    return masks_for_existing_batch(inputs["input_ids"])

attachment = attach_rope_monitor(
    trainer, config=MonitorConfig(sampling_policy="regions"),
    batch_metadata=batch_metadata,
)
```

`masks_for_existing_batch` is the caller's task-specific lookup. Its return
mapping can contain only `query_mask`, `key_mask`, and `sample_ids`. Actual
`attention_mask` and `position_ids` continue to come from the model inputs.
The callback must leave input tensors unchanged. Extra monitoring fields never
enter the model's argument list.

The plugin's seeded token selection has its own versioned rule. For byte-exact
historical paper selections, use the original saved token positions. The paper
also uses a fixed selected head collection; setting only 4Q/16K does not recreate
its complete experiment. `full_distances()` restores exhaustive positional
distances for the sampled vectors, while Gaussian reversal remains an approximation.
