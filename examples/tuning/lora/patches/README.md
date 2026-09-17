# Isolated Megatron-Bridge compatibility patches

`megatron_bridge_qwen_vl_decoder_peak.patch` targets the Qwen3-VL text
constructor used by the installed Megatron-Bridge 0.5.2 baseline. `GPTModel`
first builds a complete decoder; its VL subclass then builds another decoder
before replacing the first. Release the temporary module before evaluating
the replacement constructor, keeping initialization order, RNG draws, final
module names and tensor layouts unchanged.

Use `self.decoder = None`, not `del self.decoder`: keeping the module's
registration slot also preserves parameter traversal order. The regression
test covers both object release and this ordering contract.

The patch is in the shared VL constructor, not a model-name dispatch or a
change to the Flash-Next architecture. It is not an INT4/QAT optimization and
does not add PP, CP or VPP support.

Apply it only to an isolated dependency copy/environment, never silently to
shared site-packages. The patch paths are relative to a Python package root
containing `megatron/bridge`, not necessarily the upstream Git repository root:

```bash
patch --dry-run -p1 -d /path/to/private-package-root \
  -i /path/to/verl/examples/tuning/lora/patches/megatron_bridge_qwen_vl_decoder_peak.patch
patch -p1 -d /path/to/private-package-root \
  -i /path/to/verl/examples/tuning/lora/patches/megatron_bridge_qwen_vl_decoder_peak.patch

CUDA_VISIBLE_DEVICES='' RUN_QWEN38_BRIDGE_PATCH_TESTS=1 \
  PYTHONPATH=/path/to/private-package-root:/path/to/verl \
  python -m pytest -q tests/models/mcore/test_qwen38_next_bridge_allocation.py
```

Check the actual imported module path before using the copy. The opt-in CPU
test must fail on the affected unpatched constructor and pass after the fix;
it checks object lifetime, not measured GPU peak memory. GPU A/B must also
compare final parameter values for identical seeds/configuration. A one-layer
real-shape probe is only a construction diagnostic, not full-model loading or
GRPO acceptance. Revalidate the patch against each future Bridge release.
