# Third-party components

The HC, PLE and QSA implementations in this package are adapted from
[radixark/miles PR #2777](https://github.com/radixark/miles/pull/2777), revision
`6aea72c18bd92c35e53e7d02699a01e02de8993f`, under Apache License 2.0.
The original package is `miles_plugins/models/qwen3_8_next`.

The initial import preserves the numerical kernels and rewrites their package
namespace. Integration with NVIDIA Megatron-Bridge, the transformer-layer spec,
configuration handling and subsequent fixes are maintained here in verl.
The subsequent changes include composed mRoPE in QSA, empty packed-sequence
handling, and exception-safe PLE context hooks at the packed language boundary.
QSA additions include packed context-parallel projection/KV exchange,
rectangular query/key tensor-core kernels, and explicit mapping from packed
document blocks to physical key tiles. PLE additions include packed token
metadata reconstruction, document-clipped causal convolution halos, FP32
reverse-gradient accumulation, and context/recompute integration. SP input
gradients sum all convolution consumers before scattering back to token owners;
the original split-only backward is not used for this cross-token operation.
Provider-level CP is enabled only for separately validated configurations;
untested mixed pipeline/context topologies remain guarded.
Refer to the repository LICENSE for the Apache License 2.0 terms.
Upstream code is reference implementation evidence, not proof that the adapted
training, weight export or LoRA path has passed end-to-end validation.
