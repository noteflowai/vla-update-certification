"""Fail-closed checkpoint loading with the official architecture and remapping.

Construct directly on CUDA to avoid large transient CPU random weights. All
stored tensors are loaded strictly and compared with the checkpoint after the
same dtype conversion as the official copy-based loader. This is infrastructure,
not a model change or hardware-performance benchmark.
"""
import torch


def restore_nonpersistent_buffers(policy):
    """Rebuild small derived buffers on CPU, as in the official constructor."""
    from hashlib import sha256
    import math
    restored = {}
    stored = set(policy.state_dict())
    for prefix, module in policy.named_modules():
        local = dict(module.named_buffers(recurse=False))
        names = [name for name in local if (prefix+"."+name if prefix else name) not in stored]
        if not names:
            continue
        reference = None
        if "inv_freq" in names:
            reference = type(module)(config=module.config, device="cpu")
        for name in names:
            buffer = local[name]
            key = prefix+"."+name if prefix else name
            if reference is not None and name in ("inv_freq", "original_inv_freq"):
                expected = getattr(reference, name).to(dtype=buffer.dtype)
            elif name == "position_ids":
                expected = torch.arange(buffer.numel(), device="cpu").reshape(buffer.shape).to(buffer.dtype)
            elif name == "embed_scale" and hasattr(module, "embedding_dim"):
                expected = torch.tensor(math.sqrt(module.embedding_dim), device="cpu").to(buffer.dtype)
            else:
                raise ValueError(f"Unverified nonpersistent buffer: {key}")
            if expected.shape != buffer.shape or not torch.isfinite(expected).all():
                raise ValueError(f"Invalid derived buffer: {key}")
            module._buffers[name] = expected.to(device=buffer.device)
            if not torch.equal(module._buffers[name].cpu(), expected):
                raise RuntimeError(f"Derived buffer restoration differs: {key}")
            restored[key] = {"shape": list(expected.shape), "dtype": str(expected.dtype),
                             "cpu_reference_sha256": sha256(expected.contiguous().reshape(-1).view(
                                 torch.uint8).numpy().tobytes()).hexdigest()}
    return restored


def assign_and_verify(policy, state):
    expected = policy.state_dict()
    if set(state) != set(expected):
        raise ValueError(f"Checkpoint keys differ: missing={len(set(expected)-set(state))}, "
                         f"unexpected={len(set(state)-set(expected))}")
    for key, tensor in state.items():
        if tensor.shape != expected[key].shape:
            raise ValueError(f"Checkpoint shape differs: {key}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite checkpoint tensor: {key}")
    conversions = {key: f"{tensor.dtype}->{expected[key].dtype}"
                   for key, tensor in state.items() if tensor.dtype != expected[key].dtype}
    policy.load_state_dict(state, strict=True)
    loaded = policy.state_dict()
    for key, tensor in state.items():
        if not torch.equal(loaded[key].detach().cpu(), tensor.to(dtype=loaded[key].dtype)):
            raise RuntimeError(f"Loaded tensor differs from checkpoint: {key}")
    return {"strict_keys_and_shapes": True, "all_loaded_tensors_exact_after_official_cast": True,
            "tensor_count": len(state), "dtype_conversions": conversions}


def load_pi05(config, env_config, checkpoint):
    from lerobot.configs import FeatureType
    from lerobot.envs import env_to_policy_features
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from safetensors.torch import load_file
    from transformers.initialization import no_init_weights

    if config.type != "pi05" or config.device != "cuda":
        raise ValueError("The direct-CUDA loader is only for the declared PI05 CUDA pipeline")
    features = env_to_policy_features(env_config)
    config.output_features = {key: value for key, value in features.items()
                              if value.type is FeatureType.ACTION}
    if not config.input_features:
        config.input_features = {key: value for key, value in features.items()
                                 if key not in config.output_features}
    # Keep the default dtype and the official selective BF16/FP32 conversion.
    # Stored parameters replace all initialization; buffers use normal constructors.
    with torch.device("cuda"), no_init_weights():
        policy = PI05Policy(config)
    buffers = restore_nonpersistent_buffers(policy)
    original = load_file(str(checkpoint/"model.safetensors"), device="cpu")
    fixed = policy._fix_pytorch_state_dict_keys(original, config)
    state = {key if key.startswith("model.") else "model."+key: value
             for key, value in fixed.items()}
    audit = assign_and_verify(policy, state)
    audit.update(loader="direct_cuda_strict", architecture="official PI05Policy",
                 default_dtype=str(torch.get_default_dtype()),
                 nonpersistent_buffers_restored_from_cpu_reference=buffers)
    policy.eval()
    return policy, audit
