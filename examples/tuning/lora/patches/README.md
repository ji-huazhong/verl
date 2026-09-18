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

## Optional FlashAttention-4 import with CUTLASS DSL 4.6

`megatron_core_optional_fa4_import.patch` targets Core commit
`ba7b5ebce12af60627a80985792a1449ce45f46c` (0.18.0). FlashAttention 2.8.3
also ships a CuTe/FA4 implementation whose import can raise `AttributeError`
for `cutlass.cute.core.ThrMma` with CUTLASS DSL 4.6.2. Core probes this optional
implementation even when the selected training path does not use FA4.

The patch treats an incompatible optional import as unavailable, like a missing
module. A successful import remains enabled; unrelated exception types are not
swallowed. It does not make FA4 compatible, change an attention kernel, disable
version checks, or establish CUDA/model correctness. Do not select FA4 while
this capability is unavailable. Keep the normal FA2/TE path independently tested.

Copy the installed public Core package into an isolated namespace package root
containing `megatron/core` (it may also contain the private Bridge copy). Apply
only there, not to shared site-packages:

```bash
patch --dry-run -p1 -d /path/to/private-package-root \
  -i /path/to/verl/examples/tuning/lora/patches/megatron_core_optional_fa4_import.patch
patch -p1 -d /path/to/private-package-root \
  -i /path/to/verl/examples/tuning/lora/patches/megatron_core_optional_fa4_import.patch

CUDA_VISIBLE_DEVICES='' RUN_QWEN38_CORE_COMPAT_TESTS=1 \
  PYTHONPATH=/path/to/private-package-root:/path/to/verl \
  python -m pytest -q tests/models/mcore/test_qwen38_next_core_compat.py
```

Run the same test before applying the patch: the `AttributeError` case must
fail, while the other three pass. The test executes the actual source's import
guard with controlled imports, not the rest of Core. After patching, additionally
import Core/Bridge, verify the actual module paths and FA2 availability, and run
the CUDA/model gates. Re-audit on each dependency upgrade.
