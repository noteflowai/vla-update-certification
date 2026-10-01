"""Bind twenty future evaluation states without inspecting policy outcomes."""
import os
os.environ.setdefault("LIBERO_CONFIG_PATH", "/tmp/vlareg/libero_cfg")
os.environ.setdefault("OMP_NUM_THREADS", "2")
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
from libero.libero import benchmark, get_libero_path

BASE = Path(__file__).resolve().parent.parent
OUTPUT = BASE/"protocols/closedloop-suite-001"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    health = json.loads((BASE/"runs/pi05-health-003/manifest.json").read_text())["cases"]
    excluded_health = {(c["task"], c["init_state"]) for c in health}
    # Exclude the local first-paper pilot and additionally avoid indices 10--19.
    # This does not claim that pretrained models never saw these benchmark states.
    rng = np.random.default_rng(2026093007)
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    inventory, selected = [], []
    for task in range(suite.n_tasks):
        spec = suite.get_task(task)
        init_path = Path(get_libero_path("init_states"))/spec.problem_folder/spec.init_states_file
        bddl_path = Path(suite.get_task_bddl_file_path(task))
        states = suite.get_task_init_states(task)
        assert len(states) >= 50
        candidates = [i for i in range(20, 50) if (task, i) not in excluded_health]
        inventory.append({
            "task": task, "name": spec.name, "total_initial_states": len(states),
            "candidate_indices": candidates, "init_file": str(init_path),
            "init_file_sha256": sha256(init_path.read_bytes()).hexdigest(),
            "bddl_file": str(bddl_path), "bddl_sha256": sha256(bddl_path.read_bytes()).hexdigest(),
        })
        for index in sorted(rng.choice(candidates, size=2, replace=False).tolist()):
            state = np.asarray(states[index])
            selected.append({
                "task": task, "task_name": spec.name, "init_state": index,
                "scene_seed": 4100*10007+101*task+index, "weight": .05,
                "source_state_sha256": sha256(state.tobytes()).hexdigest(),
                "source_dtype": str(state.dtype), "source_shape": list(state.shape),
            })
    assert len(selected) == 20
    assert len({(c["task"], c["init_state"], c["scene_seed"]) for c in selected}) == 20
    assert not any((c["task"], c["init_state"]) in excluded_health for c in selected)
    record = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Future finite-suite update study; selection only, zero collected outcomes.",
        "sampling_seed": 2026093007, "candidate_range": [20, 49],
        "selection": "Two per task, without replacement, excluding local pilot and all health cases.",
        "suite": "libero_10", "task_order_index": 0, "inventory": inventory,
        "states": selected, "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "benchmark_source_sha256": sha256(Path(benchmark.__file__).read_bytes()).hexdigest(),
        "physical_reset_audit": "Required per family before outcome collection; not claimed here.",
    }
    (OUTPUT/"manifest.json").write_text(json.dumps(record, indent=2)+"\n")
    (OUTPUT/"freeze_closedloop_suite.py").write_bytes(Path(__file__).read_bytes())
    print("Twenty states bound to ten verified initial-state files and BDDL assets; zero outcomes.")


if __name__ == "__main__":
    main()
