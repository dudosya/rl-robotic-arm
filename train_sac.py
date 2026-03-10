# =============================================================================
# train_sac.py — SAC agent training for FetchReach-v3.
#
# Exports:
#   train(train_env, eval_env) -> SAC model
#   load_best_model()          -> SAC model
# =============================================================================

import os

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

from config import (
    SEED,
    TOTAL_TIMESTEPS,
    LEARNING_RATE,
    BATCH_SIZE,
    BUFFER_SIZE,
    LEARNING_STARTS,
    TAU,
    GAMMA,
    EVAL_FREQ,
    N_EVAL_TRAIN,
    MODEL_DIR,
    LOG_DIR,
)


def build_model(train_env):
    """Instantiate a fresh SAC model.

    Policy: MultiInputPolicy — handles dict observations (observation + goals).

    Key hyperparameter choices:
    - learning_rate = 1e-3       : standard for SAC on robot tasks
    - batch_size    = 256        : large batch for stable gradient estimates
    - buffer_size   = 200_000   : fits comfortably in Colab RAM
    - learning_starts = 1_000   : fill replay buffer before first update
    - gamma = 0.98              : slightly discounted — task is episode-length-50
    """
    model = SAC(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        tau=TAU,
        gamma=GAMMA,
        verbose=1,
        seed=SEED,
        tensorboard_log=os.path.join(LOG_DIR, "sac_tensorboard"),
    )
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  Policy      : {model.policy.__class__.__name__}")
    print(f"  Parameters  : {n_params:,}")
    return model


def train(train_env, eval_env):
    """Train SAC agent and save the best checkpoint.

    Uses EvalCallback to evaluate the agent every EVAL_FREQ steps on eval_env.
    The best model (by mean episode reward) is saved to MODEL_DIR/best_model.

    Returns
    -------
    model : trained SAC instance (best weights loaded)
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    print("\n" + "=" * 60)
    print("  SAC Training — FetchReach-v3 (dense reward)")
    print("=" * 60)
    print(f"  Total timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"  Eval every      : {EVAL_FREQ:,} steps  ({N_EVAL_TRAIN} episodes)")
    print(f"  Best model path : {os.path.join(MODEL_DIR, 'best_model')}")
    print("  Estimated time  : ~25–45 min on Colab T4 GPU")
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

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
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
    model = SAC.load(os.path.join(MODEL_DIR, "best_model"), env=env)
    return model
