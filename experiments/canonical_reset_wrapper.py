"""Preserve Gym vector semantics while applying the audited local reset repair."""
import gymnasium as gym
from canonical_reset import reset_without_sampler_accumulation


class CanonicalResetWrapper(gym.Wrapper):
    def reset(self, *, seed=None, options=None):
        if seed is None or options:
            raise ValueError("Canonical evaluation requires an explicit seed and no reset options")
        return reset_without_sampler_accumulation(self.env, seed)
