# Contributing

Install the source with `pip install -e '.[trainer,wandb,research,dev]'`, then run
`python -m pytest -q`. Tests build tiny random models and do not download weights
or require W&B credentials. CPU evidence is separate from CUDA evidence.

New architecture adapters must locate post-normalization, pre-RoPE Q/K and
preserve the model's native layout, rotary dimension, scaling and GQA mapping.
Check captured values against the actual forward, identical model logits,
unchanged forward counts, correct sparse storage and cleanup after exceptions.

Score changes require comparison with direct finite-window calculation on small
examples. Changes in token eligibility, distance sampling, aggregation or valid
counts must be documented and included in the exported report schema. Preserve
stable sample identities and expose any approximations in metric documentation.

Keep Trainer and logging dependencies optional. Do not introduce model loading
or account initialization into imports or monitor construction. Test built
artifacts independently of editable source installs before a release.

For installation issues include package/runtime versions and configuration.
For score issues include a synthetic reproduction and the report's counts and
frequency configuration. Raw task text or checkpoint weights are not needed.

See [release procedure](docs/RELEASING.md) and
[compatibility evidence](docs/COMPATIBILITY.md) before expanding support claims.
