# =============================================================================
# config.py — Central configuration for all hyperparameters and constants.
# Edit this file to tweak the experiment without touching any other code.
# =============================================================================

import os

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Environment ───────────────────────────────────────────────────────────────
ENV_ID            = "FetchPickAndPlace-v4"
REWARD_TYPE       = "sparse"  # sparse reward is required for HER to work correctly
MAX_EPISODE_STEPS = 100       # pick-and-place needs more steps than reach

# ── Evaluation ────────────────────────────────────────────────────────────────
N_EVAL_EPISODES = 30          # episodes per method for the final comparison

# ── Classical controllers ─────────────────────────────────────────────────────
P_GAIN = 10.0                 # proportional gain (scales Cartesian error → action)

# MuJoCo site name for the Fetch gripper end-effector
GRIP_SITE = "robot0:grip"

# Fetch Research robot — 7-DOF arm joint names (in kinematic order)
ARM_JOINT_NAMES = [
    "robot0:shoulder_pan_joint",
    "robot0:shoulder_lift_joint",
    "robot0:upperarm_roll_joint",
    "robot0:elbow_flex_joint",
    "robot0:forearm_roll_joint",
    "robot0:wrist_flex_joint",
    "robot0:wrist_roll_joint",
]

# Scipy IK: max L-BFGS-B iterations per environment step
SCIPY_MAX_ITER = 20

# ── HER (Hindsight Experience Replay) ───────────────────────────────────────────
HER_N_SAMPLED_GOAL = 4        # goals relabelled per real transition (paper default)
HER_GOAL_STRATEGY  = "future" # use a later state in the same episode as the fake goal

# ── TQC training — RL Baselines Zoo defaults for FetchPickAndPlace + HER ─────
# Reference: https://github.com/DLR-RM/rl-baselines3-zoo hyperparams/her.yml
TOTAL_TIMESTEPS = 1_000_000  # 1M is the standard benchmark target
LEARNING_RATE   = 1e-3
BATCH_SIZE      = 1024          # Zoo default for TQC+HER on FetchPickAndPlace
BUFFER_SIZE     = 1_000_000     # HER needs a larger buffer to store relabelled transitions
LEARNING_STARTS = 1_000
TAU             = 0.05
GAMMA           = 0.95
N_CRITICS       = 2             # TQC uses an ensemble of critics (n_critics=2 per Zoo)
NET_ARCH        = [512, 512, 512]  # Zoo uses [512,512,512] for TQC on pick-and-place
EVAL_FREQ       = 20_000        # evaluate every N env steps during training
N_EVAL_TRAIN    = 10            # episodes per checkpoint evaluation
CHECKPOINT_FREQ = 100_000       # save a full checkpoint every N steps (Colab crash safety)

# ── Output directories ────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
MODEL_DIR  = os.path.join(OUTPUT_DIR, "models")
LOG_DIR    = os.path.join(OUTPUT_DIR, "logs")
VIDEO_DIR  = os.path.join(OUTPUT_DIR, "videos")
PLOT_DIR   = os.path.join(OUTPUT_DIR, "plots")

# ── Control flags ─────────────────────────────────────────────────────────────
# Set SKIP_TRAINING = True to load the existing checkpoint and skip the
# ~60-min SAC training phase (useful for re-running plots/videos after training).
SKIP_TRAINING = False
