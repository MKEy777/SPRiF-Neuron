from __future__ import annotations

from collections.abc import Iterable

import torch

from .config import EvaluationConfig
from .metrics import circular_phase_error, delay_mse, event_metrics, output_radius


def select_eligible_mask(
    membrane: torch.Tensor,
    threshold: torch.Tensor | float,
    k: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if membrane.dim() != 2:
        raise ValueError("membrane must have shape [batch,hidden]")
    threshold_tensor = torch.as_tensor(
        threshold, device=membrane.device, dtype=membrane.dtype
    )
    eligible = membrane < threshold_tensor
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    order = torch.randperm(membrane.shape[1], generator=generator).to(membrane.device)
    mask = torch.zeros_like(eligible)
    valid = eligible.sum(dim=1) >= k
    for batch_index in range(membrane.shape[0]):
        ordered_eligible = order[eligible[batch_index, order]]
        chosen = ordered_eligible[:k]
        mask[batch_index, chosen] = True
    return mask, valid


def validate_invalid_rate(
    valid: torch.Tensor,
    maximum_invalid_rate: float = 0.01,
) -> None:
    invalid_rate = float((~valid).float().mean().cpu())
    if invalid_rate > maximum_invalid_rate + 1e-12:
        raise RuntimeError(
            f"invalid intervention-trial rate {invalid_rate:.2%} exceeds "
            f"the allowed {maximum_invalid_rate:.2%}"
        )


def _modes_for_model(model_name: str) -> tuple[str, ...]:
    if model_name == "sprif":
        return ("clean", "forced_no_reset", "fast_reset", "slow_reset", "both_reset")
    return ("clean", "forced_no_reset", "native_reset")


def _mean(value: torch.Tensor) -> float:
    return float(value.detach().float().mean().cpu())


@torch.no_grad()
def evaluate_paired_batch(
    model,
    inputs: torch.Tensor,
    target: torch.Tensor,
    event_step: int,
    k: int,
    gamma: float,
    margin: float,
    mask_seed: int,
    cue_steps: int,
    evaluation_cfg: EvaluationConfig,
) -> tuple[list[dict], dict]:
    model.eval()
    clean_output_all, clean_trace_all = model(inputs, mode="clean", return_trace=True)
    event_membrane = clean_trace_all["membrane_pre"][:, event_step]
    event_threshold = clean_trace_all["threshold"][:, event_step]
    mask, valid = select_eligible_mask(event_membrane, event_threshold, k, mask_seed)
    validate_invalid_rate(valid)
    valid_count = int(valid.sum().item())
    invalid_count = int((~valid).sum().item())
    if valid_count == 0:
        raise RuntimeError("no samples have enough subthreshold units for the requested K")

    inputs = inputs[valid]
    target = target[valid]
    mask = mask[valid]
    event_masks = {event_step: mask}

    outputs: dict[str, torch.Tensor] = {}
    traces: dict[str, dict[str, torch.Tensor]] = {}
    for mode in _modes_for_model(model.name):
        masks = None if mode == "clean" else event_masks
        outputs[mode], traces[mode] = model(
            inputs,
            mode=mode,
            intervention_masks=masks,
            gamma=gamma,
            margin=margin,
            return_trace=True,
        )

    forced_modes = tuple(mode for mode in _modes_for_model(model.name) if mode != "clean")
    reference_trace = traces["forced_no_reset"]
    for mode in forced_modes:
        trace = traces[mode]
        if not torch.equal(trace["spikes"][:, event_step], reference_trace["spikes"][:, event_step]):
            raise RuntimeError("forced event spike tensors differ across paired branches")
        if not torch.all(trace["forced_hit"][:, event_step][mask]):
            raise RuntimeError(f"{mode} has a missed forced spike")
        if not torch.all(trace["new_crossing"][:, event_step][mask]):
            raise RuntimeError(f"{mode} lacks a new threshold crossing")
        if model.name == "sprif":
            for key in ("spikes", "slow", "fast"):
                if not torch.equal(
                    trace[key][:, :event_step], reference_trace[key][:, :event_step]
                ):
                    raise RuntimeError(
                        f"paired SPRiF branches differ before intervention in {key}"
                    )

    reference_error = circular_phase_error(outputs["forced_no_reset"], target)
    rows = []
    sample_metrics: dict[str, dict[str, torch.Tensor]] = {}
    for mode in _modes_for_model(model.name):
        phase_error = circular_phase_error(outputs[mode], target)
        if mode == "clean":
            metric = event_metrics(
                phase_error, phase_error, event_step,
                evaluation_cfg.auc_window,
                evaluation_cfg.recovery_window,
                evaluation_cfg.recovery_sustain,
            )
        else:
            metric = event_metrics(
                phase_error, reference_error, event_step,
                evaluation_cfg.auc_window,
                evaluation_cfg.recovery_window,
                evaluation_cfg.recovery_sustain,
            )
        sample_metrics[mode] = {
            "phase_jump": metric["phase_jump"],
            "excess_auc": metric["excess_auc"],
            "recovery_time": metric["recovery_time"],
            "recovery_censored": metric["recovery_censored"],
        }
        trace = traces[mode]
        if mode == "clean":
            forced_hit_rate = 0.0
            new_crossing_rate = 0.0
            natural_spike_overlap_rate = 0.0
        else:
            selected = mask.sum().clamp_min(1)
            forced_hit_rate = float(trace["forced_hit"][:, event_step].sum().cpu() / selected.cpu())
            new_crossing_rate = float(trace["new_crossing"][:, event_step].sum().cpu() / selected.cpu())
            natural_spike_overlap_rate = float(
                trace["natural_spike"][:, event_step][mask].sum().cpu() / selected.cpu()
            )
        row = {
            "mode": mode,
            "event_step": int(event_step),
            "k": int(k),
            "gamma": float(gamma),
            "valid_samples": valid_count,
            "invalid_samples": invalid_count,
            "delay_mse": _mean(delay_mse(outputs[mode], target, cue_steps)),
            "output_radius": _mean(output_radius(outputs[mode], cue_steps)),
            "circular_phase_error": _mean(phase_error[:, cue_steps:]),
            "phase_jump": _mean(metric["phase_jump"]),
            "excess_auc": _mean(metric["excess_auc"]),
            "recovery_time": _mean(metric["recovery_time"]),
            "recovery_censored_rate": _mean(metric["recovery_censored"].float()),
            "firing_rate": _mean(trace["spikes"][:, cue_steps:]),
            "forced_hit_rate": forced_hit_rate,
            "new_crossing_rate": new_crossing_rate,
            "natural_spike_overlap_rate": natural_spike_overlap_rate,
        }
        if model.name == "sprif" and mode != "clean":
            reference_trace = traces["forced_no_reset"]
            row["slow_state_delta"] = _mean(torch.linalg.vector_norm(
                trace["slow"][:, event_step] - reference_trace["slow"][:, event_step], dim=-1
            ))
            row["fast_state_delta"] = _mean(torch.linalg.vector_norm(
                trace["fast"][:, event_step] - reference_trace["fast"][:, event_step], dim=-1
            ))
        rows.append(row)

    artifacts = {
        "valid_indices": torch.nonzero(valid, as_tuple=False).flatten().detach().cpu(),
        "mask": mask.detach().cpu(),
        "target": target.detach().cpu(),
        "outputs": {key: value.detach().cpu() for key, value in outputs.items()},
        "traces": {
            mode: {key: value.detach().cpu() for key, value in trace.items()}
            for mode, trace in traces.items()
        },
        "sample_metrics": {
            mode: {key: value.detach().cpu() for key, value in values.items()}
            for mode, values in sample_metrics.items()
        },
    }
    return rows, artifacts


@torch.no_grad()
def evaluate_multi_event_batch(
    model,
    inputs: torch.Tensor,
    target: torch.Tensor,
    event_steps: Iterable[int],
    k: int,
    gamma: float,
    margin: float,
    mask_seed: int,
    cue_steps: int,
    evaluation_cfg: EvaluationConfig,
) -> tuple[list[dict], dict]:
    if model.name != "sprif":
        raise ValueError("multi-event routing comparison is defined only for SPRiF")
    model.eval()
    event_steps = tuple(int(step) for step in event_steps)
    original_batch = inputs.shape[0]
    clean_output_all, clean_trace_all = model(inputs, mode="clean", return_trace=True)
    masks = {}
    valid = torch.ones(inputs.shape[0], device=inputs.device, dtype=torch.bool)
    for index, event_step in enumerate(event_steps):
        mask, event_valid = select_eligible_mask(
            clean_trace_all["membrane_pre"][:, event_step],
            clean_trace_all["threshold"][:, event_step],
            k,
            mask_seed + index,
        )
        masks[event_step] = mask
        valid &= event_valid
    validate_invalid_rate(valid)
    valid_count = int(valid.sum().item())
    invalid_count = int((~valid).sum().item())
    if valid_count == 0:
        raise RuntimeError("no samples remain valid across all multi-event masks")
    inputs = inputs[valid]
    target = target[valid]
    masks = {step: mask[valid] for step, mask in masks.items()}

    modes = _modes_for_model(model.name)
    outputs, traces = {}, {}
    for mode in modes:
        outputs[mode], traces[mode] = model(
            inputs,
            mode=mode,
            intervention_masks=None if mode == "clean" else masks,
            gamma=gamma,
            margin=margin,
            return_trace=True,
        )

    branch_valid = torch.ones(inputs.shape[0], device=inputs.device, dtype=torch.bool)
    forced_modes = tuple(mode for mode in modes if mode != "clean")
    for event_step, mask in masks.items():
        expected = mask.sum(dim=1)
        reference_spikes = traces["forced_no_reset"]["spikes"][:, event_step]
        for mode in forced_modes:
            crossings = (traces[mode]["new_crossing"][:, event_step] & mask).sum(dim=1)
            selected_spikes_ok = torch.all(
                (traces[mode]["spikes"][:, event_step] == 1) | ~mask, dim=1
            )
            branch_valid &= crossings == expected
            branch_valid &= selected_spikes_ok
            branch_valid &= torch.all(
                traces[mode]["spikes"][:, event_step] == reference_spikes, dim=1
            )

    final_valid = valid.clone()
    final_valid[valid] = branch_valid
    validate_invalid_rate(final_valid)
    if not torch.all(branch_valid):
        inputs = inputs[branch_valid]
        target = target[branch_valid]
        masks = {step: mask[branch_valid] for step, mask in masks.items()}
        outputs = {mode: output[branch_valid] for mode, output in outputs.items()}
        traces = {
            mode: {key: value[branch_valid] for key, value in trace.items()}
            for mode, trace in traces.items()
        }
    valid_count = int(final_valid.sum().item())
    invalid_count = int(original_batch - valid_count)
    if valid_count == 0:
        raise RuntimeError("no samples remain valid across all intervention branches")
    reference_error = circular_phase_error(outputs["forced_no_reset"], target)
    rows = []
    for mode in modes:
        phase_error = circular_phase_error(outputs[mode], target)
        excess = phase_error - reference_error
        per_event = []
        for event_step in event_steps:
            per_event.append(event_metrics(
                phase_error,
                phase_error if mode == "clean" else reference_error,
                event_step,
                evaluation_cfg.auc_window,
                evaluation_cfg.recovery_window,
                evaluation_cfg.recovery_sustain,
            ))
        if per_event:
            phase_jump = torch.stack([item["phase_jump"] for item in per_event]).mean(dim=0)
            recovery_time = torch.stack([item["recovery_time"] for item in per_event]).mean(dim=0)
            recovery_censored = torch.stack([
                item["recovery_censored"].float() for item in per_event
            ]).mean(dim=0)
        else:
            phase_jump = torch.zeros(inputs.shape[0], device=inputs.device)
            recovery_time = torch.zeros_like(phase_jump)
            recovery_censored = torch.zeros_like(phase_jump)
        selected = sum(mask.sum() for mask in masks.values())
        trace = traces[mode]
        if mode == "clean" or not event_steps:
            forced_hit_rate = 0.0
            new_crossing_rate = 0.0
            natural_spike_overlap_rate = 0.0
        else:
            hits = sum(trace["forced_hit"][:, step].sum() for step in event_steps)
            crossings = sum(trace["new_crossing"][:, step].sum() for step in event_steps)
            natural_overlap = sum(
                trace["natural_spike"][:, step][masks[step]].sum()
                for step in event_steps
            )
            forced_hit_rate = float(hits.cpu() / selected.clamp_min(1).cpu())
            new_crossing_rate = float(crossings.cpu() / selected.clamp_min(1).cpu())
            natural_spike_overlap_rate = float(
                natural_overlap.cpu() / selected.clamp_min(1).cpu()
            )
        rows.append({
            "mode": mode,
            "event_step": -1,
            "event_count": len(event_steps),
            "k": int(k),
            "gamma": float(gamma),
            "valid_samples": valid_count,
            "invalid_samples": invalid_count,
            "delay_mse": _mean(delay_mse(outputs[mode], target, cue_steps)),
            "output_radius": _mean(output_radius(outputs[mode], cue_steps)),
            "circular_phase_error": _mean(phase_error[:, cue_steps:]),
            "phase_jump": _mean(phase_jump),
            "excess_auc": _mean(excess[:, cue_steps:].sum(dim=1)),
            "recovery_time": _mean(recovery_time),
            "recovery_censored_rate": _mean(recovery_censored),
            "firing_rate": _mean(trace["spikes"][:, cue_steps:]),
            "forced_hit_rate": forced_hit_rate,
            "new_crossing_rate": new_crossing_rate,
            "natural_spike_overlap_rate": natural_spike_overlap_rate,
        })
    artifacts = {
        "mask": next(iter(masks.values())).detach().cpu() if masks else torch.zeros(
            inputs.shape[0], model.model_cfg.hidden_size, dtype=torch.bool
        ),
        "target": target.detach().cpu(),
        "outputs": {key: value.detach().cpu() for key, value in outputs.items()},
        "traces": {
            mode: {key: value.detach().cpu() for key, value in trace.items()}
            for mode, trace in traces.items()
        },
        "masks": {step: mask.detach().cpu() for step, mask in masks.items()},
    }
    return rows, artifacts
