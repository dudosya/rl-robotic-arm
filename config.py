# =============================================================================
# config.py — Central configuration for all hyperparameters and constants.
# Edit this file to tweak the experiment without touching any other code.
# =============================================================================

import os

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Environment ───────────────────────────────────────────────────────────────
ENV_ID            = "FetchPickAndPlace-v4"
REWARD_TYPE       = "dense"   # "dense" converges faster; avoids need for HER
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

# ── SAC training ──────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 500_000
LEARNING_RATE   = 1e-3
BATCH_SIZE      = 256
BUFFER_SIZE     = 500_000
LEARNING_STARTS = 2_000
TAU             = 0.005
GAMMA           = 0.98
EVAL_FREQ       = 10_000      # evaluate every N env steps during training
N_EVAL_TRAIN    = 20          # episodes per checkpoint evaluation

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
