# =============================================================================
# evaluate.py — Evaluation utilities, results reporting, and plotting.
#
# Exports:
#   evaluate_policy(env, policy_fn, n_episodes, seed) -> (mean_reward, success_rate)
#   print_results_table(results)
#   plot_training_curve(log_path, classical_results)
#   plot_comparison_bar(results)
# =============================================================================

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for Colab / headless
import matplotlib.pyplot as plt

from config import N_EVAL_EPISODES, SEED, LOG_DIR, PLOT_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_policy(env, policy_fn, n_episodes: int = N_EVAL_EPISODES, seed: int = SEED):
    """Run policy_fn for n_episodes and return summary statistics.

    Parameters
    ----------
    env        : gymnasium environment (FetchPickAndPlace-v4, dense reward)
    policy_fn  : callable(obs) -> action
    n_episodes : int
    seed       : int  (episode i uses seed + i for reproducibility)

    Returns
    -------
    mean_reward   : float
    success_rate  : float  (fraction of episodes where is_success == True)
    all_rewards   : list[float]
    all_successes : list[float]
    """
    all_rewards, all_successes = [], []

    for ep in range(n_episodes):
        obs, _       = env.reset(seed=seed + ep)
        ep_reward    = 0.0
        terminated   = truncated = False
        # Reset stateful policies (classical state machines) each episode
        if hasattr(policy_fn, "reset"):
            policy_fn.reset()

        while not (terminated or truncated):
            action                          = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

        all_rewards.append(ep_reward)
        all_successes.append(float(info.get("is_success", 0.0)))

    mean_reward  = float(np.mean(all_rewards))
    success_rate = float(np.mean(all_successes))
    return mean_reward, success_rate, all_rewards, all_successes


# ─────────────────────────────────────────────────────────────────────────────
# Console reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(results: dict):
    """Pretty-print a comparison table.

    Parameters
    ----------
    results : dict  {label: (mean_reward, success_rate)}
    """
    print("\n" + "=" * 63)
    print(f"  {'Method':<33} {'Mean Reward':>12} {'Success Rate':>12}")
    print("=" * 63)
    for label, (mean_r, succ_r) in results.items():
        print(f"  {label:<33} {mean_r:>12.3f} {succ_r:>11.1%}")
    print("=" * 63 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Training curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curve(classical_results: dict):
    """Plot SAC training reward curve with classical baselines overlaid.

    Reads the evaluations.npz log written by EvalCallback.

    Parameters
    ----------
    classical_results : dict  {label: (mean_reward, success_rate)}
                         Classical baseline values to draw as horizontal lines.
    """
    log_file = os.path.join(LOG_DIR, "evaluations.npz")
    if not os.path.exists(log_file):
        print(f"Warning: training log not found at {log_file} — skipping curve plot.")
        return

    data         = np.load(log_file)
    timesteps    = data["timesteps"]
    mean_rewards = data["results"].mean(axis=1)
    std_rewards  = data["results"].std(axis=1)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(timesteps, mean_rewards, color="steelblue", lw=2,
            label="SAC mean reward (eval)")
    ax.fill_between(timesteps,
                    mean_rewards - std_rewards,
                    mean_rewards + std_rewards,
                    alpha=0.25, color="steelblue", label="±1 std")

    # Overlay classical baselines as dashed horizontal lines
    palette = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd"]
    for (label, (mr, _)), color in zip(classical_results.items(), palette):
        ax.axhline(mr, color=color, ls="--", lw=1.5,
                   label=f"{label} ({mr:.1f})")

    ax.set_xlabel("Environment Steps", fontsize=13)
    ax.set_ylabel("Mean Episode Reward", fontsize=13)
    ax.set_title("SAC Training Curve on FetchPickAndPlace-v4 (dense reward)", fontsize=14)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, "training_curve.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_bar(results: dict):
    """Side-by-side bar charts comparing mean reward and success rate.

    Parameters
    ----------
    results : dict  {label: (mean_reward, success_rate)}
              Keys are used as x-axis method labels.
    """
    labels  = list(results.keys())
    means   = [v[0] for v in results.values()]
    succs   = [v[1] * 100 for v in results.values()]

    # Colour: RL agent (last entry) gets a distinct blue; baselines get warm tones
    n = len(labels)
    palette = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd"] + \
              ["#1f77b4"] * max(0, n - 4)
    colors  = palette[:n]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Mean reward ──────────────────────────────────────────────────
    bars1 = ax1.bar(labels, means, color=colors, edgecolor="black", linewidth=0.8)
    ax1.set_title("Mean Episode Reward", fontsize=14)
    ax1.set_ylabel("Reward", fontsize=12)
    ax1.grid(axis="y", alpha=0.3)
    span = max(means) - min(means) if max(means) != min(means) else 1.0
    for bar, val in zip(bars1, means):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + span * 0.01,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # ── Success rate ─────────────────────────────────────────────────
    bars2 = ax2.bar(labels, succs, color=colors, edgecolor="black", linewidth=0.8)
    ax2.set_title("Success Rate (%)", fontsize=14)
    ax2.set_ylabel("Success Rate (%)", fontsize=12)
    ax2.set_ylim(0, 118)
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, succs):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # Wrap long labels for readability
    for ax in [ax1, ax2]:
        ax.set_xticklabels(
            [lbl.replace(" ", "\n") for lbl in labels], fontsize=9
        )

    fig.suptitle(
        "FetchPickAndPlace-v4  |  Classical IK vs. SAC Reinforcement Learning",
        fontsize=13, fontweight="bold",
    )

    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, "comparison_chart.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")
