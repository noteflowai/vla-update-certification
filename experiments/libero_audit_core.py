"""Environment-only audit primitives; avoid importing unrelated VLA policies."""
import os
for key, value in {"MUJOCO_GL": "egl", "LIBERO_CONFIG_PATH": "/tmp/vlareg/libero_cfg",
                   "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                   "OPENBLAS_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1"}.items():
    os.environ.setdefault(key, value)
import ast
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch
from libero.libero import benchmark
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.libero import LiberoEnv

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def make_sub_env(suite, suite_name, task_id, init_state, env_cfg, episode_length):
    kwargs = {k: v for k, v in env_cfg.gym_kwargs.items() if k != "task_ids"}
    env = LiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=suite_name,
        camera_name=env_cfg.camera_name,
        episode_length=episode_length,
        episode_index=init_state,
        n_envs=1,
        control_mode=env_cfg.control_mode,
        camera_name_mapping=env_cfg.camera_name_mapping,
        **kwargs,
    )
    # Stay on this initial state on every reset.
    env._reset_stride = 0
    return env


def array_hash(value):
    value = np.asarray(value)
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": sha256(value.tobytes()).hexdigest()}


def observation_hash(value):
    if isinstance(value, dict):
        return {key: observation_hash(item) for key, item in sorted(value.items())}
    return array_hash(value)


def snapshot(env, observation):
    control = env._env
    domain = control.env
    goals = domain.parsed_problem["goal_state"]
    return {
        "state": array_hash(control.get_sim_state()),
        "observation": observation_hash(observation),
        "scene_xml_sha256": sha256(domain.sim.model.get_xml().encode()).hexdigest(),
        "prompt": env.task_description,
        "bddl_prompt": control.language_instruction,
        "goals": goals,
        "goal_values": [bool(domain._eval_predicate(goal)) for goal in goals],
        "actual_success": bool(control.check_success()),
        "controller_use_delta": [bool(robot.controller.use_delta) for robot in control.robots],
    }


def assert_reference_bodies():
    """Bind the retained constructor/snapshot bodies to the original sources."""
    def body(path, name):
        tree = ast.parse(path.read_text())
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        return ast.dump(ast.Module(body=fn.body, type_ignores=[]), include_attributes=False)
    own = Path(__file__)
    assert body(own, "make_sub_env") == body(
        Path("/home/dcvuser/work/vla-regressions/perstate_eval.py"), "make_sub_env")
    reference = own.with_name("audit_reset_scoring.py")
    for name in ("snapshot", "observation_hash", "array_hash"):
        assert body(own, name) == body(reference, name)


assert_reference_bodies()
