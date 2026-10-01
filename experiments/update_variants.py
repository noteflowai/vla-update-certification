"""Prospective update primitives, not a native inference producer or speed claim."""
from copy import deepcopy

FLOW_FIELDS = {"pi05": "num_inference_steps", "xvla": "num_denoising_steps"}
CADENCE = {"pi05": 50, "xvla": 30}
UPDATES = ("baseline_reload", "w4", "w3", "steps2")


def configure_variant(config, family, update):
    """Use the field the actual family reads; never add an unused attribute."""
    if family not in FLOW_FIELDS or update not in UPDATES or config.type != family:
        raise ValueError("Unsupported declared family or update")
    field = FLOW_FIELDS[family]
    if (getattr(config, field, None) != 10
            or config.n_action_steps != CADENCE[family]
            or config.chunk_size != CADENCE[family]):
        raise ValueError("Configuration differs from the pinned prospective pipeline")
    changed = deepcopy(config)
    if update == "steps2":
        setattr(changed, field, 2)
    if hasattr(changed, "compile_model"):
        changed.compile_model = False
    return changed, {
        "family": family, "update": update, "flow_parameter": field,
        "flow_steps": getattr(changed, field), "n_action_steps": changed.n_action_steps,
        "chunk_size": changed.chunk_size,
        "weight_bits": {"w4": 4, "w3": 3}.get(update),
        "independent_reload_required": update == "baseline_reload",
        "scope": "Configuration only; caller must independently reload the control "
                 "and apply declared rounding after strict checkpoint loading."}


def round_linear_weights(policy, update, row_chunk=128):
    """Symmetric group-128 round/dequantize; every final short group is separate."""
    if update not in ("w4", "w3") or type(row_chunk) is not int or row_chunk < 1:
        raise ValueError("Only the predeclared W4/W3 weight controls support rounding")
    import torch
    from torch import nn
    from hashlib import sha256
    bits = int(update[1])
    qmax = 2**(bits-1)-1
    seen, layers = {}, []

    def tensor_hash(value):
        digest = sha256()
        for start in range(0, value.shape[0], row_chunk):
            block = value[start:start+row_chunk].detach().contiguous().reshape(-1).view(
                torch.uint8).cpu()
            digest.update(block.numpy().tobytes())
        return digest.hexdigest()

    with torch.no_grad():
        modules = [(name, module) for name, module in policy.named_modules()
                   if isinstance(module, nn.Linear)]
        if not modules:
            raise ValueError("No linear weights in this policy")
        # Validate every selected weight before mutating any one.
        for _, module in modules:
            weight = module.weight
            if (weight.ndim != 2
                    or weight.dtype not in (torch.float32, torch.float16, torch.bfloat16)
                    or any(not torch.isfinite(weight[start:start+row_chunk]).all()
                           for start in range(0, weight.shape[0], row_chunk))):
                raise ValueError("Invalid linear weight")
        for name, module in modules:
            weight = module.weight
            if id(weight) in seen:
                layers.append({"name": name, "shared_weight_with": seen[id(weight)]})
                continue
            seen[id(weight)] = name
            before = tensor_hash(weight)
            width = weight.shape[1]
            error = 0.
            for start in range(0, weight.shape[0], row_chunk):
                block = weight[start:start+row_chunk]
                original = block.float()
                padded = torch.nn.functional.pad(original, (0, (-width) % 128))
                groups = padded.reshape(original.shape[0], -1, 128)
                scale = groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)/qmax
                grid = (groups/scale).round().clamp(-qmax, qmax)
                rounded = (grid*scale).reshape(original.shape[0], -1)[:, :width].to(weight.dtype)
                if not torch.isfinite(rounded).all():
                    raise ValueError("Nonfinite rounded weight")
                error += (rounded.float()-original).square().sum().item()
                block.copy_(rounded)
            layers.append({"name": name, "shape": list(weight.shape), "dtype": str(weight.dtype),
                           "before_sha256": before, "after_sha256": tensor_hash(weight),
                           "squared_error_sum": error, "parameters": weight.numel(),
                           "final_group_width": width % 128 or 128})
    return {"bits": bits, "group_size": 128, "row_chunk": row_chunk, "layers": layers,
            "unique_linear_weights": len(seen),
            "scope": "Floating round/dequantize, not integer-kernel execution. Shared "
                     "weights are changed once and can also be used by non-linear modules."}
