# =============================================================================
# main.py — Master script: runs the full RL vs. Classical IK experiment.
#
# Usage on Google Colab:
#   !git clone <your-repo-url>
#   %cd rl-robotic-arm
#   !pip install -r requirements.txt
#   !python main.py
#
# To skip training and reuse an existing checkpoint (re-run plots/videos):
#   Edit config.py → set SKIP_TRAINING = True, then run again.
#
# Estimated runtime on Colab T4 GPU:
#   Baseline evaluation  ~  5 min  (Scipy IK dominates)
#   SAC training         ~ 30–45 min
#   Plots + Videos       ~  3 min
#   ─────────────────────────────
#   Total                ~ 40–55 min
# =============================================================================

import os
import warnings
import numpy as np
warnings.filterwarnings("ignore")

# ── Virtual display (required for MuJoCo rendering on headless Colab) ─────────
# pyvirtualdisplay must be started BEFORE importing mujoco/gymnasium
from pyvirtualdisplay import Display
_display = Display(visible=False, size=(800, 600))
_display.start()
print("Virtual display started.")

import gymnasium
import gymnasium_robotics   # noqa: F401 — registers FetchReach-v3
import mujoco
import stable_baselines3

from stable_baselines3.common.monitor import Monitor

import config
from baselines  import RandomPolicy, PController, JacobianPinvPolicy, ScipyIKPolicy
from train_sac  import train, load_best_model
from evaluate   import evaluate_policy, print_results_table, plot_training_curve, plot_comparison_bar
from videos     import record_all

# ── Seed everything ───────────────────────────────────────────────────────────
np.random.seed(config.SEED)

# ── Print version info ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RL vs. Classical IK for Robotic Arm Control")
print("  FetchReach-v3 (MuJoCo)")
print("=" * 60)
print(f"  MuJoCo              : {mujoco.__version__}")
print(f"  gymnasium-robotics  : {gymnasium_robotics.__version__}")
print(f"  stable-baselines3   : {stable_baselines3.__version__}")
print(f"  SKIP_TRAINING       : {config.SKIP_TRAINING}")
print(f"  SEED                : {config.SEED}")
print(f"  TOTAL_TIMESTEPS     : {config.TOTAL_TIMESTEPS:,}")
print("=" * 60 + "\n")

os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.PLOT_DIR,   exist_ok=True)
os.makedirs(config.VIDEO_DIR,  exist_ok=True)
os.makedirs(config.MODEL_DIR,  exist_ok=True)
os.makedirs(config.LOG_DIR,    exist_ok=True)


# =============================================================================
# 1. CLASSICAL BASELINE EVALUATION
# =============================================================================
print("─" * 60)
print("  Step 1/4 — Evaluating classical baselines")
print("─" * 60)

def make_env():
    """Create a standard FetchReach-v3 env (no render mode)."""
    return gymnasium.make(
        config.ENV_ID,
        reward_type=config.REWARD_TYPE,
        max_episode_steps=config.MAX_EPISODE_STEPS,
    )

classical_results = {}   # {label: (mean_reward, success_rate)}

# ── Random ────────────────────────────────────────────────────────────────────
print("\n[1/4] Random Policy ...")
env = make_env()
policy = RandomPolicy(env)
mr, sr, _, _ = evaluate_policy(env, policy)
env.close()
classical_results["Random Policy"] = (mr, sr)
print(f"  mean_reward={mr:.3f}  success_rate={sr:.1%}")

# ── Proportional Cartesian Controller ────────────────────────────────────────
print("\n[2/4] Proportional Cartesian Controller ...")
env = make_env()
policy = PController(env)
mr, sr, _, _ = evaluate_policy(env, policy)
env.close()
classical_results["P-Controller"] = (mr, sr)
print(f"  mean_reward={mr:.3f}  success_rate={sr:.1%}")

# ── Jacobian Pseudo-Inverse ───────────────────────────────────────────────────
print("\n[3/4] Jacobian Pseudo-Inverse ...")
env = make_env()
policy = JacobianPinvPolicy(env)
mr, sr, _, _ = evaluate_policy(env, policy)
env.close()
classical_results["Jacobian Pseudo-Inv"] = (mr, sr)
print(f"  mean_reward={mr:.3f}  success_rate={sr:.1%}")

# ── Scipy Numerical IK ────────────────────────────────────────────────────────
print("\n[4/4] Scipy Numerical IK (L-BFGS-B) — slower, please wait ...")
env = make_env()
policy = ScipyIKPolicy(env)
mr, sr, _, _ = evaluate_policy(env, policy)
env.close()
classical_results["Scipy IK (L-BFGS-B)"] = (mr, sr)
print(f"  mean_reward={mr:.3f}  success_rate={sr:.1%}")

print("\nClassical baselines done.")
print_results_table(classical_results)


# =============================================================================
# 2. SAC TRAINING (or load existing checkpoint)
# =============================================================================
print("─" * 60)
if config.SKIP_TRAINING:
    print("  Step 2/4 — Loading existing SAC checkpoint (SKIP_TRAINING=True)")
    print("─" * 60)
    sac_model = load_best_model()
else:
    print("  Step 2/4 — Training SAC agent")
    print("─" * 60)
    train_env = Monitor(make_env())
    eval_env  = Monitor(make_env())
    sac_model = train(train_env, eval_env)
    train_env.close()
    eval_env.close()


# =============================================================================
# 3. PLOT TRAINING CURVE
# =============================================================================
print("\n─" * 60)
print("  Step 3/4 — Generating plots")
print("─" * 60)

plot_training_curve(classical_results)   # reads outputs/logs/evaluations.npz


# =============================================================================
# 4. EVALUATE SAC + FULL COMPARISON
# =============================================================================
print("\n[SAC evaluation] Running 30 episodes with best checkpoint ...")
env = make_env()
sac_policy = lambda obs: sac_model.predict(obs, deterministic=True)[0]
mr, sr, _, _ = evaluate_policy(env, sac_policy)
env.close()
print(f"  mean_reward={mr:.3f}  success_rate={sr:.1%}")

all_results = dict(classical_results)
all_results["SAC (150k steps)"] = (mr, sr)

print_results_table(all_results)
plot_comparison_bar(all_results)


# =============================================================================
# 5. DEMO VIDEOS
# =============================================================================
print("─" * 60)
print("  Step 4/4 — Recording demo videos")
print("─" * 60 + "\n")

# Ready policies are passed directly.
# Jacobian / Scipy policies need an env reference to access model/data, so we
# pass factory lambdas as (factory, True) tuples — videos.py handles this.
named_policies = {
    "random":       RandomPolicy,       # will be called as factory(env)
    "p_controller": PController,
    "jacobian_ik":  JacobianPinvPolicy,
    "scipy_ik":     ScipyIKPolicy,
    "sac_trained":  lambda env: (lambda obs: sac_model.predict(obs, deterministic=True)[0]),
}
# Wrap each as a factory tuple so record_all applies them to fresh render envs
factory_dict = {k: (v, True) for k, v in named_policies.items()}
record_all(factory_dict)


# =============================================================================
# DONE
# =============================================================================
print("\n" + "=" * 60)
print("  All done!")
print("=" * 60)
print(f"\nOutputs written to: ./{config.OUTPUT_DIR}/")
print(f"  Plots   : {config.PLOT_DIR}/")
print(f"    - training_curve.png")
print(f"    - comparison_chart.png")
print(f"  Videos  : {config.VIDEO_DIR}/")
print(f"    - demo_random.mp4")
print(f"    - demo_p_controller.mp4")
print(f"    - demo_jacobian_ik.mp4")
print(f"    - demo_scipy_ik.mp4")
print(f"    - demo_sac_trained.mp4")
print(f"  Models  : {config.MODEL_DIR}/best_model.zip")
print("")
