"""Local, version-bound reset repair; no installed package files are modified."""
from hashlib import sha256
import ast
import inspect
import os
from pathlib import Path
import textwrap
os.environ.setdefault("LIBERO_CONFIG_PATH", "/tmp/vlareg/libero_cfg")
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from libero.libero.envs.bddl_base_domain import BDDLBaseDomain
from libero.libero.envs.regions.object_property_sampler import OpenCloseSampler, TurnOnOffSampler

BASE_SHA = "4f4da47dd241ac6590d66c7559d76e44b68669924207f53143ca3a4962921f24"


def verified_sampler_builder(method):
    """Accept the base builder or an exact, argument-free super delegation."""
    function = method.__func__
    if function is BDDLBaseDomain._add_placement_initializer:
        return True
    mro = type(method.__self__).__mro__
    owner = next((index for index, cls in enumerate(mro)
                  if cls.__dict__.get("_add_placement_initializer") is function), None)
    if owner is None:
        return False
    delegated = next((cls.__dict__["_add_placement_initializer"] for cls in mro[owner+1:]
                      if "_add_placement_initializer" in cls.__dict__), None)
    if delegated is not BDDLBaseDomain._add_placement_initializer:
        return False
    body = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0].body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    expected = ast.parse("super()._add_placement_initializer()").body
    return (function.__name__ == "_add_placement_initializer"
            and ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)
            == ast.dump(ast.Module(body=expected, type_ignores=[]), include_attributes=False))


def clear_generated_property_samplers(domain):
    samplers = domain.object_property_initializers
    if not isinstance(samplers, list) or any(type(item) not in (
            OpenCloseSampler, TurnOnOffSampler) for item in samplers):
        raise ValueError("Unknown/custom property sampler; refuse to clear it")
    count = len(samplers)
    samplers.clear()
    return count


def reset_without_sampler_accumulation(env, seed):
    """Regenerate one set of default samplers on each declared hard reset."""
    source = Path(inspect.getfile(BDDLBaseDomain))
    if sha256(source.read_bytes()).hexdigest() != BASE_SHA:
        raise ValueError("Unverified LIBERO reset implementation")
    if env.hard_reset is not True:
        raise ValueError("The repair only supports the audited hard-reset pipeline")
    np.random.seed(seed)
    env._ensure_env()
    domain = env._env.env
    if domain.hard_reset is not True or not verified_sampler_builder(domain._add_placement_initializer):
        raise ValueError("Unverified property-sampler construction")
    before = clear_generated_property_samplers(domain)
    observation, info = env.reset(seed=seed)
    expected = sum(state[0] in ("open", "close", "turnon", "turnoff")
                   and state[1] in domain.object_states_dict
                   and hasattr(domain.object_states_dict[state[1]], "set_joint")
                   for state in domain.parsed_problem["initial_state"])
    after = len(domain.object_property_initializers)
    if after != expected:
        raise RuntimeError("Property samplers were not regenerated exactly once")
    info = {**info, "reset_repair": {"generated_samplers_before": before,
                                   "generated_samplers_after": after,
                                   "expected_generated_samplers": expected,
                                   "libero_base_sha256": BASE_SHA,
                                   "task_domain_source_sha256": sha256(
                                       Path(inspect.getfile(type(domain))).read_bytes()).hexdigest()}}
    return observation, info
