"""Temporary H20 validation harness for MTP plus fused main-head Linear CE.

Run from the repository root. This process artifact can be removed once the
feature is production-ready.
"""

from __future__ import annotations

import argparse
import os
import runpy
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def _reference_outputs(logits: torch.Tensor, labels: torch.Tensor, temperature: float):
    logits = logits.float() / temperature
    log_probs = logits.log_softmax(dim=-1)
    selected = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return selected.reshape(-1), entropy.reshape(-1)


class _OutputLayer(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.clone())
        self.weight.grad_added_to_main_grad = False
        self.weight.main_grad = torch.zeros_like(self.weight)
        self.bias = None

    def forward(self, hidden_states, weight=None, **_kwargs):
        return F.linear(hidden_states, self.weight if weight is None else weight), None


class _SyntheticMTPModel(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.output_layer = _OutputLayer(weight)
        self.share_embeddings_and_output_weights = False
        self.post_process = True
        self.config = SimpleNamespace(
            mtp_num_layers=1,
            mtp_loss_scaling_factor=0.2,
            mtp_detach_heads=False,
            calculate_per_token_loss=False,
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            use_mup=False,
            fp8=None,
        )

    @staticmethod
    def compute_language_model_loss(labels, logits):
        flat_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1), reduction="none")
        return flat_loss.view_as(labels)


def run_mtp_integration() -> None:
    from verl.models.mcore import model_forward_fused as fused
    from verl.models.mcore import mtp_patch

    if not mtp_patch._HAS_PROCESS_MTP_LOSS:
        raise RuntimeError("H20 integration test requires native MCore process_mtp_loss")

    fused.parallel_state.get_tensor_model_parallel_group = lambda: None
    torch.manual_seed(20260919)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    tokens, hidden_size, vocab_size = 384, 512, 32768
    temperature = 0.7
    labels = torch.randint(0, vocab_size, (1, tokens), device=device)
    loss_mask = torch.ones_like(labels, dtype=torch.float32)
    loss_mask[:, -1] = 0
    base_weight = torch.randn(vocab_size, hidden_size, device=device, dtype=dtype) * 0.02
    base_hidden = torch.randn(tokens * 2, 1, hidden_size, device=device, dtype=dtype) * 0.02
    results = []

    for use_fused in (False, True):
        model = _SyntheticMTPModel(base_weight).to(device)
        # Metric logging is orthogonal to the gradient contract under test and
        # requires a fully initialized Megatron DP/CP topology.
        model.eval()
        hidden = base_hidden.clone().requires_grad_()
        kwargs = {}
        if use_fused:
            kwargs = {
                "output_processor": fused.fused_output_processor,
                "output_processor_context": fused.FusedOutputProcessorContext(temperature, labels),
            }
        output = mtp_patch._megatron_gptmodel_postprocess(
            model,
            hidden_states=hidden,
            input_ids=labels,
            position_ids=None,
            labels=labels,
            rotary_pos_emb=None,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            loss_mask=loss_mask.clone(),
            **kwargs,
        )
        if use_fused:
            log_probs, entropy = output.log_probs, output.entropy
            if output.logits is not None:
                raise AssertionError("fused main head materialized logits")
            if not model.output_layer.weight.zero_out_wgrad:
                raise AssertionError("mixed MCore/autograd wgrad contract was not enabled")
        else:
            log_probs, entropy = _reference_outputs(output, labels, temperature)
        loss = -log_probs.mean() - 0.01 * entropy.mean()
        loss.backward()
        results.append(
            {
                "log_probs": log_probs.detach().float(),
                "entropy": entropy.detach().float(),
                "hidden_grad": hidden.grad.detach().float(),
                "weight_grad": model.output_layer.weight.grad.detach().float(),
            }
        )

    tolerances = {
        "log_probs": (2e-3, 5e-4),
        "entropy": (8e-3, 8e-4),
        "hidden_grad": (2e-2, 4e-2),
        "weight_grad": (2e-2, 4e-2),
    }
    for name, (atol, rtol) in tolerances.items():
        reference, actual = results[0][name], results[1][name]
        torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
        print(
            f"H20_MTP_FUSED {name} max_abs={(actual - reference).abs().max().item():.8g} "
            f"reference_norm={reference.norm().item():.8g}"
        )
    print("H20_MTP_FUSED integration=PASS native_auxiliary=true actual_triton=true")


def run_tp2_kernel() -> None:
    script = Path("tests/utils/test_special_linear_cross_entropy_tp.py")
    namespace = runpy.run_path(str(script), run_name="h20_tp_benchmark")
    cls = namespace["TestLinearCrossEntropy_TensorParallel"]
    cls.generate_hyper.__globals__["MAX_TEST_CASES"] = 1
    torch.manual_seed(233376 + int(os.environ["RANK"]))
    test = cls()
    try:
        test.initialize(0)
        test.check_torch_storage()
        test.verify_kernel_correctness(iterations=3)
        test.check_kernel_storage()
    finally:
        test.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("mtp", "tp2"))
    args = parser.parse_args()
    if args.mode == "mtp":
        run_mtp_integration()
    else:
        run_tp2_kernel()


if __name__ == "__main__":
    main()
