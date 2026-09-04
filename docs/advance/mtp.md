# Guide to Using MTP in SFT/RL Training and Inference

**Author**: `https://github.com/meituan-search`

Last updated: 09/04/2026

## 1. Scope of Support

Currently, RL training can be performed on mimo-7B-RL, Qwen-next, and Deepseek series models based on the MTP architecture. The support rules for training and inference engines are as follows:

- **Training Engine**: Only supports the `mbridge/Megatron-Bridge + megatron` combination; other training engines are not compatible at this time;

- **Inference Engine**: Compatible with all engines, but the model must be in the corresponding engine's compatibility list;

- **Dependency Versions**:

    - mbridge: Apply the patches and review suggestions from PR: [#62](https://github.com/ISEEKYAN/mbridge/pull/62) (Already merged into the main branch);

    - Megatron-Bridge: Apply the patches and review suggestions from PR if you want to try out mimo-7B-RL: [#2387](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2387) (will be merged into the main branch in the future);

    - megatron: Use the latest dev version (commit: [23e092f41ec8bc659020e401ddac9576c1cfed7e](https://github.com/NVIDIA/Megatron-LM/tree/23e092f41ec8bc659020e401ddac9576c1cfed7e)), which supports MTP + CP training methods. If you additionally enable `recompute_granularity=full`, use a dev commit that includes [#3457](https://github.com/NVIDIA/Megatron-LM/pull/3457) (`ffd66a3e6`, 2026-06-03) — the commit pinned above predates it by about six months. #3457 threads `padding_mask` through `MultiTokenPredictionLayer._checkpointed_forward`; without it, `MultiTokenPredictionLayer.forward` passes a keyword the method does not declare and training raises a `TypeError` on the first step. Released `megatron-core` 0.18.0 and 0.18.2 do not carry the fix either (tracked as [#4933](https://github.com/NVIDIA/Megatron-LM/issues/4933)).
    
    - sglang: Use the specified branch: [https://github.com/ArronHZG/sglang/tree/fix_mtp_update_weights_from_tensor](https://github.com/ArronHZG/sglang/tree/fix_mtp_update_weights_from_tensor), [PR](https://github.com/sgl-project/sglang/pull/17870) , which fix the MTP update weights from tensor OOM issue.

## 2. MTP Training Configuration (Core Parameters)

The MTP training process can be flexibly controlled through the following configurations. All configurations are based on the `actor_rollout_ref.model.mtp` prefix:

| Configuration Scenario | Core Parameters                                                                                                                                                                                                                                                                                               | Description                                             |
|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| Load MTP Parameters Only | `enable=True`                                                                                                                                                                                                                                                                                              | VRAM usage will increase, but the exported parameters include the MTP module and can be directly used for online deployment              |
| Full-Parameter MTP Training | `enable=True`<br>`enable_train=True`<br>`mtp_loss_scaling_factor=0.1`                                                                                                                                                                                                                              | MTP Loss will apply to all model parameters                            |
| MTP Parameter-Only Training | `enable=True`<br>`enable_train=True`<br>`detach_encoder=True`                                                                                                                                                                                                                                      | Freeze the Encoder layer, update only MTP module parameters, MTP Loss applies only to MTP parameters |
| MTP Accelerated Rollout | 1. vLLM configuration:<br>`enable=True`<br>`enable_rollout=True`<br>`method="mtp"`<br>`num_speculative_tokens=1`<br>2. SGLang configuration:<br>`enable=True`<br>`enable_rollout=True`<br>`speculative_algorithm="EAGLE"`<br>`speculative_num_steps=2`<br>`speculative_eagle_topk=2`<br>`speculative_num_draft_tokens=4` | Achieve inference acceleration during the Rollout phase based on MTP                      |

### Memory-efficient main output head (Megatron Engine)

`actor_rollout_ref.model.use_fused_kernels=True` can be used together with MTP
in the Megatron Engine. This fuses **only the main output head** with linear
cross entropy. MTP auxiliary heads still use the existing MCore/legacy logits
and CE path, including their original weight-gradient, masking, logging and
loss-normalization behavior. No new kernel or checkpoint format is introduced.
For MCore DDP, the output weight uses `zero_out_wgrad=True` so that normal
autograd gradients from the fused main head are added alongside auxiliary
gradients accumulated directly into `main_grad`; otherwise the main gradient
can be skipped by DDP's `grad_added_to_main_grad` check.

```yaml
actor_rollout_ref:
  model:
    use_remove_padding: true
    use_fused_kernels: true
    mtp:
      enable: true
      enable_train: true  # false: load MTP without computing auxiliary loss
```

The integration requires a text GPTModel with the native `output_processor`
and `output_processor_context` forward contract (MCore 0.18 or a compatible
build), remove-padding/THD, and sequence parallelism when TP > 1. Value models,
vision wrappers, legacy forward replacements, output-layer bias, FP8 output
projection, deferred output-layer weight gradients and MuP logit scaling remain
unsupported and disable fused execution with a warning. The
existing CUDA/Triton kernel requirements still apply. Nonuniform temperatures,
top-K distillation and `calculate_sum_pi_squared` are not supported with fused
execution; disable `use_fused_kernels` for these workloads.

Main-head labels are passed separately from the model's MTP labels, so
`enable_train=False` does not accidentally activate auxiliary loss. Training
passes the same response mask alignment, packed positions and Dynamic CP loss
normalization metadata as the non-fused path.

Before enabling this in production, compare fixed-batch runs with fused
execution off/on on the target GPUs: main log probabilities/entropy, each MTP
loss, parameter gradients, peak allocated memory and step time. This first
stage does **not** eliminate auxiliary-head logits, and memory savings do not
imply a throughput improvement. Further auxiliary Linear CE optimization should
be driven by these measurements, not enabled by forcing detached output weights.

Regression commands (run in an environment with the corresponding dependencies):

```bash
# CPU contracts with explicitly stubbed collectives/kernel; needs PyTorch + pytest.
pytest -q tests/models/test_mtp_fused_main_ce_on_cpu.py

# Additional tests in a provisioned verl + Megatron environment.
pytest -q tests/models/test_model_forward_fused.py tests/utils/test_megatron_mtp_dcp.py \
  tests/workers/test_megatron_mtp_fused_gate.py
```

## 3. Experimental Results

The experiment was conducted as follows:

* model = mimo-7B-math
* max_response_length = 8k

Experiment chart:

![fully_async_policy_revenue](
https://github.com/ArronHZG/verl-community/blob/main/docs/mimo-7b-mtp.png?raw=true)

The wandb link for the graph: [wandb](https://wandb.ai/hou-zg-meituan/mimo-7b-sft-mtp?nw=nwuserhouzg)

**Scenarios with No Significant Effect**

The following configurations will not have a noticeable impact on training results:

1. The base model does not carry MTP parameters;

2. The base model carries MTP parameters, but the MTP module is not trained;

3. The base model carries MTP parameters and trains MTP, with `mtp_loss_scaling_factor=0`;

4. The base model carries MTP parameters, trains MTP and detaches the encoder, with `mtp_loss_scaling_factor=0.1`.

**Scenarios with Significant Effect**

Only the following configuration will have a noticeable impact on training results:

- The base model carries MTP parameters, MTP Loss applies to all model parameters, and `mtp_loss_scaling_factor=0.1`.

**Recommended Training Method**

It is recommended to adopt the `detach_encoder=True` approach for MTP training.

## 4. Performance Notes for MTP in Rollout Inference

Enabling MTP improves the rollout acceptance rate by around 14%. However, on H20 GPUs, overall throughput does not increase and even decreases slightly.

![spec_log](
https://github.com/ArronHZG/verl-community/blob/main/docs/spec_log.png?raw=true)

The effectiveness of MTP-accelerated Rollout is significantly affected by **model size** and **inference hardware**. Key reference information is as follows:

**Hardware Tensor Core Performance**

| Hardware Model | FP16 Performance (TFLOPS) |
|----------------|---------------------------|
| H20  | 148            |
| H800 | 1,671          |
| H200 | 1,979          |

**Measured Performance and Recommendations**

Taking the mimo-7B model deployed separately on H20 hardware using SGLang as an example: After enabling MTP speculative decoding, the Rollout throughput decreases by approximately 50%.

- Current priority recommendation: Do not enable MTP acceleration during the inference phase for now;

- Future planning: Further optimization of the speculative logic in the Rollout phase will be conducted to improve throughput performance.

## 5. SFT training

The SFT training with MTP is supported, using the same MTP training configuration as RL training.

An example configuration for running SFT can be found in `examples/sft/gsm8k/run_mimo_7b_mtp_megatron.sh`

**SFT result**

The experiment was conducted using following data:
- model = mimo-7B-math
- dataset = gsm8k

The result: [wandb link](https://wandb.ai/hou-zg-meituan/mimo-7b-sft-mtp?nw=nwuserhouzg)

The presence of mtp layer has limited effect on main loss. However, when MTP layer is detached, the mtp_loss converges to a higher value.
