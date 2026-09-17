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
Refer to the repository LICENSE for the Apache License 2.0 terms.
Upstream code is reference implementation evidence, not proof that the adapted
training, weight export or LoRA path has passed end-to-end validation.
