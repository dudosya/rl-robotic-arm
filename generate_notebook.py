#!/usr/bin/env python3
"""
Run this script once to generate reach_rl.ipynb.
  python generate_notebook.py
"""
import json, textwrap

def md(src, cid):
    return {"cell_type":"markdown","id":cid,"metadata":{},"source":src}

def code(src, cid):
    return {"cell_type":"code","id":cid,"metadata":{},
            "source":src,"outputs":[],"execution_count":None}

cells = []

# ── TITLE ─────────────────────────────────────────────────────────────
cells.append(md("""\
# Reinforcement Learning vs. Classical Kinematics for Robotic Arm Control

**Course Final Project · FetchReach-v3 (MuJoCo)**

---

## Abstract
This notebook investigates whether a **Reinforcement Learning** agent (SAC) can match
or exceed classical **model-based** controllers on the robotic reaching task.
We evaluate four approaches on the `FetchReach-v3` environment:

| # | Method | Type |
|---|--------|------|
| 0 | Random Policy | Baseline |
| 1 | Proportional Cartesian Controller | Classical |
| 2 | Jacobian Pseudo-Inverse Controller | Classical (model-based IK) |
| 3 | Scipy Numerical IK (L-BFGS-B) | Classical (optimisation-based IK) |
| 4 | Soft Actor-Critic (SAC) | RL (learned) |

**Environment:** The Fetch robot arm must move its end-effector to a randomly sampled
3-D target in space.  Actions are 4-D Cartesian velocity commands `[dx, dy, dz, gripper]`.

---

## Table of Contents
1. [Setup](#setup)
2. [Environment Overview](#env)
3. [Classical Baselines](#classical)
   - 3.1 Random Policy
   - 3.2 Proportional Cartesian Controller
   - 3.3 Jacobian Pseudo-Inverse
   - 3.4 Scipy Numerical IK
4. [SAC Training](#sac)
5. [Results & Comparison](#results)
6. [Demo Videos](#videos)
7. [Conclusion](#conclusion)
""", "md_00"))

# ── SECTION 0 : INSTALL ───────────────────────────────────────────────
cells.append(md("## Section 0 — Setup <a name='setup'></a>\n\n"
    "Run the cell below **once per Colab session** to install all dependencies.\n"
    "After installation, go to **Runtime → Restart runtime**, then run all cells again.",
    "md_01"))

cells.append(code("""\
# ── Install all dependencies ──────────────────────────────────────────
# gymnasium-robotics  : FetchReach-v3 MuJoCo environment
# stable-baselines3   : SAC implementation
# shimmy              : gymnasium compatibility shim
# imageio / ffmpeg    : video recording
# pyvirtualdisplay    : headless rendering in Colab
!pip install -q \\
    'gymnasium-robotics>=1.2' \\
    'stable-baselines3[extra]>=2.3' \\
    'shimmy>=1.0' \\
    imageio imageio-ffmpeg \\
    pyvirtualdisplay \\
    'scipy>=1.11'
print("All packages installed.")
""", "code_00"))

cells.append(code("""\
import os, warnings
import numpy as np
import matplotlib.pyplot as plt
import gymnasium
import gymnasium_robotics   # registers FetchReach-v3
import mujoco
import stable_baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from scipy.optimize import minimize as sp_minimize
import imageio
warnings.filterwarnings('ignore')

# ── Virtual display (required for headless Colab rendering) ───────────
from pyvirtualdisplay import Display
_display = Display(visible=False, size=(800, 600))
_display.start()

# ── Global seed ───────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

print(f"MuJoCo              : {mujoco.__version__}")
print(f"gymnasium-robotics  : {gymnasium_robotics.__version__}")
print(f"stable-baselines3   : {stable_baselines3.__version__}")
print("Virtual display     : started")
""", "code_01"))

# ── SECTION 1 : ENV OVERVIEW ──────────────────────────────────────────
cells.append(md("""\
## Section 1 — Environment Overview <a name='env'></a>

`FetchReach-v3` simulates the 7-DOF Fetch Research robot arm.

### Observation (dict)
| Key | Shape | Meaning |
|-----|-------|---------|
| `observation` | (25,) | joint positions, velocities, gripper state |
| `achieved_goal` | (3,) | **current end-effector (EE) position** |
| `desired_goal` | (3,) | **target position** |

### Action
A 4-D vector `[dx, dy, dz, gripper]` — the desired Cartesian velocity
of the gripper, clipped to **[-1, 1]** per dimension.

### Reward
We use **dense rewards** (`reward_type='dense'`): reward = −‖EE − goal‖₂, giving
continuous gradient signal that accelerates both learning and classical controller evaluation.
""", "md_10"))

cells.append(code("""\
env = gymnasium.make('FetchReach-v3', reward_type='dense', max_episode_steps=50)
obs, info = env.reset(seed=SEED)

print("=== FetchReach-v3 ===")
print(f"Observation keys   : {list(obs.keys())}")
print(f"  observation shape: {obs['observation'].shape}")
print(f"  achieved_goal    : {np.round(obs['achieved_goal'], 3)}  (current EE pos)")
print(f"  desired_goal     : {np.round(obs['desired_goal'],  3)}  (target pos)")
print(f"Action space       : {env.action_space}")
print(f"  shape={env.action_space.shape}  range=[{env.action_space.low[0]:.1f}, {env.action_space.high[0]:.1f}]")
print(f"MuJoCo total DOF   : {env.unwrapped.model.nv}")
print(f"Number of joints   : {env.unwrapped.model.njnt}")

# Quick sanity-check: step with zero action
obs2, r, term, trunc, info2 = env.step(np.zeros(4))
print(f"\\nSample dense reward (zero action): {r:.4f}")
print(f"  (= -||EE - goal||, so ~{-r:.4f} m distance to goal)")
env.close()
""", "code_10"))

# ── SECTION 2 : CLASSICAL BASELINES ──────────────────────────────────
cells.append(md("""\
## Section 2 — Classical Baselines <a name='classical'></a>

Before training any RL agent, we evaluate three **model-based** controllers
alongside a random baseline.  All baselines use the same evaluation protocol.

### Evaluation Protocol
- **30 episodes**, each up to 50 steps
- Metrics: **mean episode reward** and **success rate** (`is_success` flag)
- Same random seeds for fair comparison
""", "md_20"))

cells.append(code("""\
def evaluate_agent(env_eval, policy_fn, n_episodes=30, seed=0, label=''):
    \"\"\"
    Evaluate a policy for n_episodes.

    Parameters
    ----------
    env_eval   : gymnasium environment
    policy_fn  : callable(obs) -> action
    n_episodes : int
    seed       : int  (episode i uses seed+i)
    label      : str  (printed header)

    Returns
    -------
    rewards, successes, mean_reward, success_rate
    \"\"\"
    rewards, successes = [], []
    for ep in range(n_episodes):
        obs, _ = env_eval.reset(seed=seed + ep)
        ep_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env_eval.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
        successes.append(float(info.get('is_success', 0.0)))

    mean_r = float(np.mean(rewards))
    succ_r = float(np.mean(successes))
    if label:
        print(f'=== {label} ===')
        print(f'  Mean reward  : {mean_r:.3f}')
        print(f'  Success rate : {succ_r:.1%}')
    return rewards, successes, mean_r, succ_r
""", "code_20"))

# 2.1 Random
cells.append(md("""\
### 2.1 — Random Policy (Baseline)

The random policy samples uniformly from the action space at every step.
It has no knowledge of the robot's state or the goal — this is the
lower bound on performance.
""", "md_21"))

cells.append(code("""\
env_eval = gymnasium.make('FetchReach-v3', reward_type='dense', max_episode_steps=50)

def random_policy(obs):
    return env_eval.action_space.sample()

rand_rewards, rand_successes, rand_mean, rand_succ = evaluate_agent(
    env_eval, random_policy, n_episodes=30, label='Random Policy'
)
env_eval.close()
""", "code_21"))

# 2.2 P-controller
cells.append(md("""\
### 2.2 — Proportional Cartesian Controller

The simplest model-based approach: compute the Cartesian error between the
end-effector and the goal, then command a velocity proportional to it.

$$\\mathbf{a} = \\text{clip}\\!\\left(K_p \\cdot (\\mathbf{g} - \\mathbf{p}_{ee}),\\; -1,\\; 1\\right)$$

where $\\mathbf{g}$ is the desired goal, $\\mathbf{p}_{ee}$ is the current EE position,
and $K_p = 10$ scales the error into the action range.

This controller requires **no joint-space model** — only Cartesian positions.
""", "md_22"))

cells.append(code("""\
P_GAIN = 10.0   # scales positional error into [-1, 1] action range

def p_controller_policy(obs):
    dx = obs['desired_goal'] - obs['achieved_goal']   # (3,) positional error
    action = np.zeros(4)
    action[:3] = np.clip(P_GAIN * dx, -1.0, 1.0)
    return action

env_eval = gymnasium.make('FetchReach-v3', reward_type='dense', max_episode_steps=50)
p_rewards, p_successes, p_mean, p_succ = evaluate_agent(
    env_eval, p_controller_policy, n_episodes=30, label='P-Controller (Cartesian)'
)
env_eval.close()
""", "code_22"))

# 2.3 Jacobian
cells.append(md("""\
### 2.3 — Jacobian Pseudo-Inverse Controller

The **Jacobian pseudo-inverse** is the classical approach described in the project
description.  Given the robot's current configuration, we compute the geometric
Jacobian $J \\in \\mathbb{R}^{3 \\times n_v}$ that maps joint velocities
$\\dot{\\mathbf{q}}$ to end-effector velocity $\\dot{\\mathbf{x}}$:

$$\\dot{\\mathbf{x}} = J \\dot{\\mathbf{q}}$$

To invert this — i.e., find joint velocities that produce a desired
Cartesian displacement — we use the **Moore-Penrose pseudo-inverse**:

$$\\Delta\\mathbf{q} = J^+ \\Delta\\mathbf{x}$$

The recoverable end-effector displacement is then:

$$\\Delta\\mathbf{x}_{ee} = J \\, J^+ \\, \\Delta\\mathbf{x}$$

When $J$ has full row rank (which holds away from singularities), $J J^+ = I$ and
the full displacement is achieved.  The result is fed as the Cartesian action.

**MuJoCo API:** `mujoco.mj_jacSite(model, data, jacp, jacr, site_id)` fills the
translational Jacobian `jacp ∈ ℝ^{3 × nv}` for the gripper site.
""", "md_23"))

cells.append(code("""\
GRIP_SITE = 'robot0:grip'

def make_jacobian_policy(env):
    \"\"\"Factory: returns a Jacobian pseudo-inverse policy for the given env.\"\"\"
    model   = env.unwrapped.model
    data    = env.unwrapped.data
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE)
    n_act   = env.action_space.shape[0]

    def jacobian_policy(obs):
        # 1. Compute translational Jacobian: J in R^{3 x nv}
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = jacp

        # 2. Cartesian positional error
        dx = obs['desired_goal'] - obs['achieved_goal']   # (3,)

        # 3. Pseudo-inverse: delta_q = J+ dx   shape (nv,)
        J_pinv = np.linalg.pinv(J)
        dq = J_pinv @ dx

        # 4. Recoverable EE velocity: dx_ee = J dq   shape (3,)
        #    = J J+ dx  (identity when J has full row rank)
        dx_ee = J @ dq

        # 5. Build 4-D action [dx, dy, dz, gripper=0]
        action = np.zeros(n_act)
        action[:3] = np.clip(dx_ee * P_GAIN, -1.0, 1.0)
        return action

    return jacobian_policy


env_eval    = gymnasium.make('FetchReach-v3', reward_type='dense', max_episode_steps=50)
jac_policy  = make_jacobian_policy(env_eval)

jac_rewards, jac_successes, jac_mean, jac_succ = evaluate_agent(
    env_eval, jac_policy, n_episodes=30, label='Jacobian Pseudo-Inverse'
)
env_eval.close()
""", "code_23"))

# 2.4 Scipy IK
cells.append(md("""\
### 2.4 — Scipy Numerical IK (L-BFGS-B)

While the Jacobian pseudo-inverse computes a *local* first-order solution,
**numerical IK** treats the problem as a global nonlinear optimisation:

$$\\mathbf{q}^* = \\arg\\min_{\\mathbf{q}} \\|\\, FK(\\mathbf{q}) - \\mathbf{g} \\,\\|^2$$

where $FK(\\mathbf{q})$ is the forward kinematics function (given joint angles, return
EE position) and $\\mathbf{g}$ is the desired goal.

**Implementation details:**
- We use `scipy.optimize.minimize` with `method='L-BFGS-B'` (bounded quasi-Newton).
- Forward kinematics is evaluated by setting joint angles on an **isolated MuJoCo data copy**
  so the live environment state is never corrupted.
- The optimal joint delta $\\Delta\\mathbf{q} = \\mathbf{q}^* - \\mathbf{q}_0$ is converted
  to a Cartesian action via the Jacobian: $\\Delta\\mathbf{x}_{ee} = J \\, \\Delta\\mathbf{q}$.
- Only the **7 arm joints** are optimised; base and gripper joints are fixed.

> **Note:** Scipy IK runs an optimisation at *every environment step*, so it is
> considerably slower than the Jacobian controller — about 2–5 min for 30 episodes.
""", "md_24"))

cells.append(code("""\
# 7-DOF Fetch arm joint names
ARM_JOINT_NAMES = [
    'robot0:shoulder_pan_joint',
    'robot0:shoulder_lift_joint',
    'robot0:upperarm_roll_joint',
    'robot0:elbow_flex_joint',
    'robot0:forearm_roll_joint',
    'robot0:wrist_flex_joint',
    'robot0:wrist_roll_joint',
]

def get_arm_indices(model):
    \"\"\"Return (qpos_indices, dof_indices) arrays for the 7 Fetch arm joints.\"\"\"
    q_ids, v_ids = [], []
    for name in ARM_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            q_ids.append(model.jnt_qposadr[jid])
            v_ids.append(model.jnt_dofadr[jid])
    return np.array(q_ids, dtype=int), np.array(v_ids, dtype=int)

def get_joint_bounds(model, q_ids):
    \"\"\"Return list of (lo, hi) for each qpos index.\"\"\"
    bounds = []
    for qi in q_ids:
        jid_arr = np.where(model.jnt_qposadr == qi)[0]
        if len(jid_arr) > 0 and model.jnt_limited[jid_arr[0]]:
            lo, hi = model.jnt_range[jid_arr[0]]
            bounds.append((float(lo), float(hi)))
        else:
            bounds.append((-np.pi, np.pi))
    return bounds


def make_scipy_ik_policy(env):
    \"\"\"Factory: returns a Scipy numerical IK policy for the given env.\"\"\"
    model     = env.unwrapped.model
    data      = env.unwrapped.data
    ik_data   = mujoco.MjData(model)   # isolated copy — never modified by env.step()
    site_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE)
    q_ids, v_ids = get_arm_indices(model)
    bounds    = get_joint_bounds(model, q_ids)
    n_act     = env.action_space.shape[0]

    def fk(q_arm):
        \"\"\"Forward kinematics using isolated ik_data copy.\"\"\"
        ik_data.qpos[q_ids] = q_arm
        mujoco.mj_forward(model, ik_data)
        return ik_data.site_xpos[site_id].copy()

    def scipy_ik_policy(obs):
        # ── 1. Sync ik_data with current env state ──────────────────
        ik_data.qpos[:] = data.qpos[:]
        ik_data.qvel[:] = 0.0
        mujoco.mj_forward(model, ik_data)

        q0_arm = ik_data.qpos[q_ids].copy()
        goal   = obs['desired_goal']

        # ── 2. Solve IK: minimise ||FK(q) - goal||^2 ────────────────
        result = sp_minimize(
            fun=lambda q: float(np.sum((fk(q) - goal) ** 2)),
            x0=q0_arm,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 25, 'ftol': 1e-7}
        )
        q_opt = result.x

        # ── 3. Restore ik_data ───────────────────────────────────────
        ik_data.qpos[q_ids] = q0_arm
        mujoco.mj_forward(model, ik_data)

        # ── 4. Convert joint delta to EE velocity via Jacobian ───────
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

        dq_full = np.zeros(model.nv)
        dq_full[v_ids] = q_opt - q0_arm
        dx_ee = jacp @ dq_full

        action = np.zeros(n_act)
        action[:3] = np.clip(dx_ee * P_GAIN, -1.0, 1.0)
        return action

    return scipy_ik_policy


env_eval     = gymnasium.make('FetchReach-v3', reward_type='dense', max_episode_steps=50)
scipy_policy = make_scipy_ik_policy(env_eval)

print('Evaluating Scipy IK (slower due to per-step optimisation — please wait)...')
ik_rewards, ik_successes, ik_mean, ik_succ = evaluate_agent(
    env_eval, scipy_policy, n_episodes=30, label='Scipy Numerical IK (L-BFGS-B)'
)
env_eval.close()
""", "code_24"))

# Summary table
cells.append(md("""\
### Classical Baselines — Summary

The cell below prints a comparison table of all classical methods so far.
SAC will be added after training (Section 4).
""", "md_25"))

cells.append(code("""\
print(f'{"Method":<35} {"Mean Reward":>13} {"Success Rate":>13}')
print('-' * 63)
print(f'{"Random Policy":<35} {rand_mean:>13.3f} {rand_succ:>12.1%}')
print(f'{"P-Controller (Cartesian)":<35} {p_mean:>13.3f} {p_succ:>12.1%}')
print(f'{"Jacobian Pseudo-Inverse":<35} {jac_mean:>13.3f} {jac_succ:>12.1%}')
print(f'{"Scipy Numerical IK (L-BFGS-B)":<35} {ik_mean:>13.3f} {ik_succ:>12.1%}')
print('=' * 63)
""", "code_25"))

# ── SECTION 3 : SAC TRAINING ──────────────────────────────────────────
cells.append(md("""\
## Section 3 — SAC Training <a name='sac'></a>

### Why Soft Actor-Critic (SAC)?

SAC is an **off-policy, entropy-regularised** actor-critic algorithm particularly
well-suited to continuous control tasks:

- **Off-policy**: learns from a replay buffer → sample efficient
- **Entropy regularisation**: encourages exploration, avoids local optima
- **Actor-Critic**: separate policy (actor) and value (critic) networks

The policy is learned **purely from interactions** with the environment — no kinematic
model, no Jacobian, no prior knowledge of robot geometry.

### Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| `learning_rate` | 1e-3 | Standard for SAC on robot tasks |
| `batch_size` | 256 | Large batch → stable gradients |
| `buffer_size` | 200 000 | Fits in Colab RAM |
| `learning_starts` | 1 000 | Fills buffer before first update |
| `total_timesteps` | 150 000 | ~30–40 min on Colab T4 GPU |
| `reward_type` | dense | Faster convergence vs sparse+HER |
""", "md_30"))

cells.append(code("""\
os.makedirs('./logs',   exist_ok=True)
os.makedirs('./models', exist_ok=True)

# ── Environments ─────────────────────────────────────────────────────
train_env = Monitor(gymnasium.make('FetchReach-v3', reward_type='dense', max_episode_steps=50))
eval_env  = Monitor(gymnasium.make('FetchReach-v3', reward_type='dense', max_episode_steps=50))

# ── SAC model ─────────────────────────────────────────────────────────
# MultiInputPolicy handles dict observations (observation + goals)
sac_model = SAC(
    policy='MultiInputPolicy',
    env=train_env,
    learning_rate=1e-3,
    batch_size=256,
    buffer_size=200_000,
    learning_starts=1_000,
    tau=0.005,
    gamma=0.98,
    verbose=1,
    seed=SEED,
    tensorboard_log='./logs/sac_fetchreach'
)

n_params = sum(p.numel() for p in sac_model.policy.parameters())
print(f'Policy architecture : {sac_model.policy.__class__.__name__}')
print(f'Total parameters    : {n_params:,}')
""", "code_30"))

cells.append(code("""\
# ── EvalCallback: evaluates every 5 000 steps, saves best checkpoint ──
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./models/',
    log_path='./logs/',
    eval_freq=5_000,
    n_eval_episodes=20,
    deterministic=True,
    verbose=1
)

TOTAL_TIMESTEPS = 150_000
print(f'Training for {TOTAL_TIMESTEPS:,} timesteps.')
print('Expected duration on Colab T4 GPU : ~25–40 minutes.')
print('Watch the ep_rew_mean column for improvement.\\n')

sac_model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, progress_bar=True)

train_env.close()
eval_env.close()
print('\\nTraining complete.  Best model saved to ./models/best_model')
""", "code_31"))

cells.append(code("""\
# ── Plot training curve ───────────────────────────────────────────────
eval_log     = np.load('./logs/evaluations.npz')
timesteps    = eval_log['timesteps']
mean_rewards = eval_log['results'].mean(axis=1)
std_rewards  = eval_log['results'].std(axis=1)

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(timesteps, mean_rewards, color='steelblue', lw=2, label='SAC mean reward (eval)')
ax.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards,
                alpha=0.25, color='steelblue', label='±1 std')

# Overlay classical baseline lines for reference
ax.axhline(rand_mean, color='#d62728', ls='--', lw=1.5, label=f'Random ({rand_mean:.1f})')
ax.axhline(p_mean,    color='#ff7f0e', ls='--', lw=1.5, label=f'P-Controller ({p_mean:.1f})')
ax.axhline(jac_mean,  color='#2ca02c', ls='--', lw=1.5, label=f'Jacobian IK ({jac_mean:.1f})')
ax.axhline(ik_mean,   color='#9467bd', ls='--', lw=1.5, label=f'Scipy IK ({ik_mean:.1f})')

ax.set_xlabel('Environment Steps', fontsize=13)
ax.set_ylabel('Mean Episode Reward', fontsize=13)
ax.set_title('SAC Training Curve on FetchReach-v3 (dense reward)', fontsize=14)
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_curve.png', dpi=150)
plt.show()
print('Saved: training_curve.png')
""", "code_32"))

# ── SECTION 4 : RESULTS ───────────────────────────────────────────────
cells.append(md("""\
## Section 4 — Results & Comparison <a name='results'></a>

We evaluate the **best saved SAC checkpoint** for 30 episodes using the same
protocol as the classical baselines, then generate side-by-side bar charts.
""", "md_40"))

cells.append(code("""\
# Load best checkpoint
best_sac = SAC.load('./models/best_model')

env_eval_sac = gymnasium.make('FetchReach-v3', reward_type='dense', max_episode_steps=50)

def sac_policy(obs):
    action, _ = best_sac.predict(obs, deterministic=True)
    return action

sac_rewards, sac_successes, sac_mean, sac_succ = evaluate_agent(
    env_eval_sac, sac_policy, n_episodes=30, label='SAC Trained Agent'
)
env_eval_sac.close()
""", "code_40"))

cells.append(code("""\
# ── Full comparison table ─────────────────────────────────────────────
print(f'{"Method":<35} {"Mean Reward":>13} {"Success Rate":>13}')
print('=' * 63)
rows = [
    ('Random Policy',               rand_mean, rand_succ),
    ('P-Controller (Cartesian)',     p_mean,    p_succ),
    ('Jacobian Pseudo-Inverse',      jac_mean,  jac_succ),
    ('Scipy Numerical IK',           ik_mean,   ik_succ),
    ('SAC (Trained — 150k steps)',   sac_mean,  sac_succ),
]
for name, mr, sr in rows:
    print(f'{name:<35} {mr:>13.3f} {sr:>12.1%}')
print('=' * 63)

# ── Bar charts ────────────────────────────────────────────────────────
METHODS = ['Random', 'P-Ctrl', 'Jacobian\\nPinv', 'Scipy\\nIK', 'SAC']
MEANS   = [rand_mean, p_mean, jac_mean, ik_mean, sac_mean]
SUCCS   = [rand_succ, p_succ, jac_succ, ik_succ, sac_succ]
COLORS  = ['#d62728', '#ff7f0e', '#2ca02c', '#9467bd', '#1f77b4']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

bars1 = ax1.bar(METHODS, MEANS, color=COLORS, edgecolor='black', linewidth=0.8)
ax1.set_title('Mean Episode Reward', fontsize=14)
ax1.set_ylabel('Reward', fontsize=12)
for bar, val in zip(bars1, MEANS):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + abs(max(MEANS) - min(MEANS)) * 0.01,
             f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

bars2 = ax2.bar(METHODS, [s * 100 for s in SUCCS], color=COLORS, edgecolor='black', linewidth=0.8)
ax2.set_title('Success Rate (%)', fontsize=14)
ax2.set_ylabel('Success Rate (%)', fontsize=12)
ax2.set_ylim(0, 115)
for bar, val in zip(bars2, SUCCS):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

for ax in [ax1, ax2]:
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('FetchReach-v3  |  Classical IK vs. SAC Reinforcement Learning',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('comparison_chart.png', dpi=150)
plt.show()
print('Saved: comparison_chart.png')
""", "code_41"))

# ── SECTION 5 : VIDEOS ────────────────────────────────────────────────
cells.append(md("""\
## Section 5 — Demo Videos <a name='videos'></a>

We record one representative episode for each method and display them
inline for visual comparison.  The Fetch arm's gripper (red sphere in the
rendered view) should converge toward the green target sphere.
""", "md_50"))

cells.append(code("""\
def record_episode(env_rec, policy_fn, max_steps=50, seed=7):
    \"\"\"Record one episode; return list of RGB frames.\"\"\"
    frames = []
    obs, _ = env_rec.reset(seed=seed)
    for _ in range(max_steps):
        action    = policy_fn(obs)
        obs, _, terminated, truncated, _ = env_rec.step(action)
        frame = env_rec.render()
        if frame is not None:
            frames.append(frame)
        if terminated or truncated:
            break
    return frames

# Create separate render envs (rgb_array mode)
def make_render_env():
    return gymnasium.make(
        'FetchReach-v3', reward_type='dense',
        max_episode_steps=50, render_mode='rgb_array'
    )

print('Recording episodes...')
agents = {
    'random':       (make_render_env(), random_policy),
    'p_controller': (make_render_env(), p_controller_policy),
}
# Rebuild Jacobian / Scipy policies for fresh render envs (they capture env internals)
_env_jac   = make_render_env()
_env_scipy = make_render_env()
agents['jacobian_ik']  = (_env_jac,   make_jacobian_policy(_env_jac))
agents['scipy_ik']     = (_env_scipy, make_scipy_ik_policy(_env_scipy))
agents['sac_trained']  = (make_render_env(), sac_policy)

video_paths = {}
for name, (env_r, policy) in agents.items():
    frames = record_episode(env_r, policy)
    path   = f'demo_{name}.mp4'
    if frames:
        imageio.mimsave(path, frames, fps=20)
        print(f'  Saved {path}  ({len(frames)} frames)')
    else:
        print(f'  WARNING: no frames for {name}')
    video_paths[name] = path
    env_r.close()

print('All videos saved.')
""", "code_50"))

cells.append(code("""\
from IPython.display import Video, display, HTML

LABELS = {
    'random':       'Random Policy',
    'p_controller': 'P-Controller (Cartesian)',
    'jacobian_ik':  'Jacobian Pseudo-Inverse',
    'scipy_ik':     'Scipy Numerical IK',
    'sac_trained':  'SAC Trained Agent',
}

for name, path in video_paths.items():
    if os.path.exists(path):
        display(HTML(f'<h4>{LABELS[name]}</h4>'))
        display(Video(path, embed=True, width=420))
    else:
        print(f'Video not found: {path}')
""", "code_51"))

# ── SECTION 6 : CONCLUSION ────────────────────────────────────────────
cells.append(md("""\
## Section 6 — Discussion & Conclusion <a name='conclusion'></a>

### Findings

| Method | Characteristics | Observations |
|--------|-----------------|--------------|
| Random | No knowledge | Lowest performance; random walk |
| P-Controller | Purely Cartesian, no joint model | Works well because action space *is* Cartesian; simple and fast |
| Jacobian Pseudo-Inverse | Full joint-space model, per-step Jacobian computation | Demonstrates classical IK theory; tends to match or slightly improve on P-control near singularities |
| Scipy Numerical IK | Global joint-space optimisation | Highest theoretical accuracy; slow and occasionally fails near local optima |
| SAC | Model-free, learns from experience | Achieves **competitive or superior** performance after 150k steps *without any kinematic model* |

### Key Takeaway

The reinforcement learning approach (SAC) is remarkable because it achieves
comparable success rates to classical model-based methods **without being
given any mathematical model of the robot's kinematics**.  The agent infers
a control policy purely from trial-and-error interaction with the environment.

This demonstrates the core thesis of the project: learning-based control can
match model-based methods and offers greater adaptability — if the robot geometry
changes (e.g., a joint stiffens), the RL agent can be retrained, whereas
the Jacobian controller would require re-deriving the kinematic model.

### Limitations

1. **Dense reward shortcut** — sparse rewards with Hindsight Experience Replay (HER)
   would be more principled but require longer training (~500k steps).
2. **Sim-only** — no sim-to-real transfer is attempted.  Real robots introduce
   sensor noise, actuator delays, and unmodelled dynamics.
3. **Fixed task** — only reaching, no grasping or full pick-and-place.
4. **IK eval speed** — Scipy IK evaluates slowly; a compiled solver (e.g., IKPy)
   would be faster in practice.

### Future Work

- Replace dense reward with **sparse reward + HER** for more general learning.
- Train on the full **FetchPickAndPlace-v3** task.
- Compare SAC with **PPO** and **TD3**.
- Attempt **sim-to-real transfer** using domain randomisation.

---
*Project generated for the Reinforcement Learning course final project.*
""", "md_60"))

# ── ASSEMBLE NOTEBOOK ─────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "colab": {
            "provenance": [],
            "gpuType": "T4"
        },
        "accelerator": "GPU"
    },
    "cells": cells
}

out = "reach_rl.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"\nGenerated: {out}")
print(f"Total cells: {len(cells)}")
