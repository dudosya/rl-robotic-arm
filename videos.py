# =============================================================================
# videos.py — Record demonstration episodes and save as .mp4 files.
#
# Exports:
#   record_episode(env, policy_fn, seed) -> list[np.ndarray]
#   save_video(frames, path)
#   record_all(policies, seed)
# =============================================================================

import os

import numpy as np
import imageio
import gymnasium

from config import (
    ENV_ID,
    REWARD_TYPE,
    MAX_EPISODE_STEPS,
    SEED,
    VIDEO_DIR,
)


def record_episode(env, policy_fn, max_steps: int = MAX_EPISODE_STEPS, seed: int = SEED):
    """Run one episode and collect RGB frames.

    Parameters
    ----------
    env        : gymnasium environment with render_mode='rgb_array'
    policy_fn  : callable(obs) -> action
    max_steps  : int  (safety cap)
    seed       : int

    Returns
    -------
    frames : list of (H, W, 3) uint8 arrays
    """
    frames = []
    obs, _ = env.reset(seed=seed)
    if hasattr(policy_fn, "reset"):
        policy_fn.reset()

    for _ in range(max_steps):
        action               = policy_fn(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if terminated or truncated:
            break

    return frames


def save_video(frames: list, path: str, fps: int = 8):
    """Write a list of RGB frames to an mp4 file.

    Parameters
    ----------
    frames : list of (H, W, 3) uint8 arrays
    path   : output filepath (should end in .mp4)
    fps    : frames per second
    """
    if not frames:
        print(f"  Warning: no frames to save for {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)
    print(f"  Saved: {path}  ({len(frames)} frames @ {fps} fps)")


def record_all(named_policies: dict, seed: int = SEED + 7):
    """Record one demo episode per policy and save to VIDEO_DIR.

    Each policy needs its own render env because the Jacobian/Scipy policies
    capture environment internals (model/data references).  We create a fresh
    render env for each policy, pass it to the factory if needed, record, then
    close.

    Parameters
    ----------
    named_policies : dict
        Keys   : short name used in the output filename  (e.g. 'sac_trained')
        Values : either:
                   - callable(obs) -> action           (policy already built)
                   - callable(env) -> callable(obs)    (factory that needs env)
                 We detect factories by checking if the value requires an env
                 argument — pass a tuple (factory, True) to flag a factory.

    seed : int  (used for all recordings)

    Returns
    -------
    saved_paths : dict  {name: filepath}
    """
    os.makedirs(VIDEO_DIR, exist_ok=True)
    saved_paths = {}

    for name, policy_entry in named_policies.items():
        print(f"  Recording: {name} ...")

        # Create a fresh render environment for this policy
        render_env = gymnasium.make(
            ENV_ID,
            reward_type=REWARD_TYPE,
            max_episode_steps=MAX_EPISODE_STEPS,
            render_mode="rgb_array",
        )

        # Determine if entry is a (factory_fn, True) tuple or a ready policy
        if isinstance(policy_entry, tuple):
            factory_fn, _ = policy_entry
            policy_fn     = factory_fn(render_env)
        else:
            policy_fn = policy_entry

        frames = record_episode(render_env, policy_fn, seed=seed)
        render_env.close()

        out_path = os.path.join(VIDEO_DIR, f"demo_{name}.mp4")
        save_video(frames, out_path)
        saved_paths[name] = out_path

    print(f"\nAll demo videos saved to: {VIDEO_DIR}/")
    return saved_paths
