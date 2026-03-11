# =============================================================================
# train_sac.py — TQC+HER agent training for FetchPickAndPlace-v4.
#
# Exports:
#   train(train_env, eval_env) -> TQC model
#   load_best_model()          -> TQC model
# =============================================================================

import os

from sb3_contrib import TQC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

from config import (
    SEED,
    TOTAL_TIMESTEPS,
    LEARNING_RATE,
    BATCH_SIZE,
    BUFFER_SIZE,
    LEARNING_STARTS,
    TAU,
    GAMMA,
    N_CRITICS,
    NET_ARCH,
    EVAL_FREQ,
    N_EVAL_TRAIN,
    CHECKPOINT_FREQ,
    MODEL_DIR,
    LOG_DIR,
    HER_N_SAMPLED_GOAL,
    HER_GOAL_STRATEGY,
)


def build_model(train_env):
    """Instantiate a fresh TQC+HER model.

    Why TQC instead of SAC?
    SAC on FetchPickAndPlace learns to *push* the block along the table, which
    succeeds on the 50% of episodes where the goal is at table height — but it
    never grasps or lifts the block.  TQC (Truncated Quantile Critics) uses a
    distributional critic ensemble that provides richer gradient signal,
    allowing the agent to learn the full pick-and-place sequence.  It is the
    only algorithm the rl-baselines3-zoo benchmarks on this task.

    Key hyperparameter choices (from rl-baselines3-zoo her.yml):
    - n_critics = 2              : ensemble of 2 critics (TQC default)
    - net_arch = [512, 512, 512] : 3-layer 512-wide network
    - batch_size = 1024          : larger batch for distributional targets
    - n_sampled_goal = 4         : HER goals per real transition
    - goal_selection_strategy    : 'future' — later states as hindsight goals
    """
    model = TQC(
        policy="MultiInputPolicy",
        env=train_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": HER_N_SAMPLED_GOAL,
            "goal_selection_strategy": HER_GOAL_STRATEGY,
        },
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        tau=TAU,
        gamma=GAMMA,
        policy_kwargs={"n_critics": N_CRITICS, "net_arch": NET_ARCH},
        verbose=1,
        seed=SEED,
        tensorboard_log=os.path.join(LOG_DIR, "tqc_tensorboard"),
    )
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  Policy      : {model.policy.__class__.__name__}")
    print(f"  Parameters  : {n_params:,}")
    return model


def train(train_env, eval_env):
    """Train TQC agent and save the best checkpoint.

    Uses EvalCallback to evaluate every EVAL_FREQ steps and CheckpointCallback
    to save a full snapshot every CHECKPOINT_FREQ steps.  The checkpoint files
    protect against Colab session disconnects — if training is interrupted, load
    the latest tqc_checkpoint_XXXXXX_steps.zip from outputs/models/ manually.

    Returns
    -------
    model : trained TQC instance (best weights loaded)
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    print("\n" + "=" * 60)
    print("  TQC + HER Training — FetchPickAndPlace-v4 (sparse reward)")
    print("=" * 60)
    print(f"  Total timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"  HER n_sampled   : {HER_N_SAMPLED_GOAL}  strategy={HER_GOAL_STRATEGY!r}")
    print(f"  Eval every      : {EVAL_FREQ:,} steps  ({N_EVAL_TRAIN} episodes)")
    print(f"  Best model path : {os.path.join(MODEL_DIR, 'best_model')}")
    print(f"  Checkpoint every: {CHECKPOINT_FREQ:,} steps → {MODEL_DIR}/tqc_checkpoint_*.zip")
    print("  Estimated time  : ~2.5–3 h on Colab T4 GPU")
    print("=" * 60 + "\n")

    model = build_model(train_env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_TRAIN,
        deterministic=True,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=MODEL_DIR,
        name_prefix="tqc_checkpoint",
        verbose=1,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=CallbackList([eval_callback, checkpoint_callback]),
        progress_bar=True,
    )

    print(f"\nTraining complete.  Best model saved to: {MODEL_DIR}/best_model")

    # Return model with best weights loaded
    return load_best_model(train_env)


def load_best_model(env=None):
    """Load the best saved SAC checkpoint.

    Parameters
    ----------
    env : optional gymnasium env
        If provided, re-attaches the environment for further training.
        For inference-only use, env=None is fine.
    """
    checkpoint = os.path.join(MODEL_DIR, "best_model.zip")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Run with SKIP_TRAINING=False first to train the agent."
        )
    print(f"Loading best model from: {checkpoint}")
    model = TQC.load(os.path.join(MODEL_DIR, "best_model"), env=env)
    return model
