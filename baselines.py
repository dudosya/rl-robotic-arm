# =============================================================================
# baselines.py — Classical (model-based) baseline policies for FetchReach-v3.
#
# Four policies are implemented:
#   1. RandomPolicy          — uniformly random actions (lower-bound reference)
#   2. PController           — proportional Cartesian controller
#   3. JacobianPinvPolicy    — geometric Jacobian pseudo-inverse (IK)
#   4. ScipyIKPolicy         — numerical IK via scipy L-BFGS-B optimisation
#
# Each policy is a callable:  action = policy(obs)
# where obs is the FetchReach-v3 dict observation and action is a (4,) ndarray.
# =============================================================================

import numpy as np
import mujoco
from scipy.optimize import minimize as sp_minimize

from config import (
    P_GAIN,
    GRIP_SITE,
    ARM_JOINT_NAMES,
    SCIPY_MAX_ITER,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Random Policy
# ─────────────────────────────────────────────────────────────────────────────

class RandomPolicy:
    """Samples uniformly from the action space at every step.

    Serves as the uninformed lower-bound baseline — equivalent to a robot with
    no sensor feedback or kinematic model.
    """

    def __init__(self, env):
        self.action_space = env.action_space

    def __call__(self, obs):
        return self.action_space.sample()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Proportional Cartesian Controller
# ─────────────────────────────────────────────────────────────────────────────

class PController:
    """Proportional (P) controller in Cartesian space.

    Computes the error between the end-effector (EE) position and the goal,
    then outputs a velocity command proportional to that error:

        a = clip(K_p * (goal - ee_pos), -1, 1)

    Because FetchReach-v3 actions ARE Cartesian velocities, this controller
    requires no joint-space model — only the Cartesian positions provided
    directly in the observation dictionary.
    """

    def __init__(self, env, gain: float = P_GAIN):
        self.gain     = gain
        self.n_action = env.action_space.shape[0]

    def __call__(self, obs):
        dx     = obs["desired_goal"] - obs["achieved_goal"]   # (3,) error
        action = np.zeros(self.n_action)
        action[:3] = np.clip(self.gain * dx, -1.0, 1.0)
        return action


# ─────────────────────────────────────────────────────────────────────────────
# Shared MuJoCo helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_arm_indices(model):
    """Return (qpos_indices, dof_indices) for the 7 Fetch arm joints."""
    q_ids, v_ids = [], []
    for name in ARM_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            q_ids.append(int(model.jnt_qposadr[jid]))
            v_ids.append(int(model.jnt_dofadr[jid]))
    return np.array(q_ids, dtype=int), np.array(v_ids, dtype=int)


def _get_joint_bounds(model, q_ids):
    """Return list of (lo, hi) bounds for each qpos index."""
    bounds = []
    for qi in q_ids:
        match = np.where(model.jnt_qposadr == qi)[0]
        if len(match) > 0 and model.jnt_limited[match[0]]:
            lo, hi = model.jnt_range[match[0]]
            bounds.append((float(lo), float(hi)))
        else:
            bounds.append((-np.pi, np.pi))
    return bounds


# ─────────────────────────────────────────────────────────────────────────────
# 3. Jacobian Pseudo-Inverse Controller
# ─────────────────────────────────────────────────────────────────────────────

class JacobianPinvPolicy:
    """Jacobian pseudo-inverse controller (differential/first-order IK).

    At each time step:
      1. Compute the translational Jacobian J ∈ R^{3 × nv} for the gripper site
         using mujoco.mj_jacSite().
      2. Compute the Cartesian positional error Δx = goal − ee_pos.
      3. Compute joint-velocity command:  Δq = J†  Δx   (Moore-Penrose pinv)
      4. Recover EE velocity:             Δx_ee = J Δq   (= Δx when J full-rank)
      5. Clip and scale into the [-1, 1] action range.

    This is the standard model-based IK approach described in the project
    description (Jacobian-based differential kinematics).
    """

    def __init__(self, env, gain: float = P_GAIN):
        self.model   = env.unwrapped.model
        self.data    = env.unwrapped.data
        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE
        )
        self.n_action = env.action_space.shape[0]
        self.gain     = gain

    def __call__(self, obs):
        # 1. Translational Jacobian:  J ∈ R^{3 × nv}
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        J = jacp

        # 2. Cartesian positional error
        dx = obs["desired_goal"] - obs["achieved_goal"]   # (3,)

        # 3. Pseudo-inverse:  Δq = J† Δx
        J_pinv = np.linalg.pinv(J)
        dq     = J_pinv @ dx

        # 4. Recoverable EE velocity:  Δx_ee = J Δq
        #    Equals Δx when J has full row rank (no singularity)
        dx_ee = J @ dq

        # 5. Build 4-D action  [dx, dy, dz, gripper=0]
        action = np.zeros(self.n_action)
        action[:3] = np.clip(dx_ee * self.gain, -1.0, 1.0)
        return action


# ─────────────────────────────────────────────────────────────────────────────
# 4. Scipy Numerical IK Policy
# ─────────────────────────────────────────────────────────────────────────────

class ScipyIKPolicy:
    """Numerical IK controller using scipy L-BFGS-B optimisation.

    Unlike the Jacobian pseudo-inverse (which gives a local, first-order
    solution), numerical IK formulates a global optimisation problem:

        q* = argmin_q  || FK(q) − goal ||²

    where FK(q) is the forward kinematics function — given joint angles q,
    return the end-effector Cartesian position.

    Implementation details:
    - A separate, isolated MjData copy is used for FK evaluation so the live
      environment state is never corrupted by the optimiser's trial evaluations.
    - Only the 7 arm joints are optimised; all other DOFs are held fixed.
    - The optimal joint delta Δq = q* − q₀ is converted to a Cartesian action
      via the Jacobian:  Δx_ee = J Δq.
    - Per-step optimisation makes this controller considerably slower than the
      Jacobian approach (~3–6 min for 30 episodes on Colab).
    """

    def __init__(self, env, gain: float = P_GAIN):
        self.model   = env.unwrapped.model
        self.data    = env.unwrapped.data
        # Isolated MjData copy — only used inside the FK function
        self.ik_data = mujoco.MjData(self.model)

        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE
        )
        self.q_ids, self.v_ids = _get_arm_indices(self.model)
        self.bounds   = _get_joint_bounds(self.model, self.q_ids)
        self.n_action = env.action_space.shape[0]
        self.gain     = gain

    def _fk(self, q_arm):
        """Forward kinematics: set arm joints, run mj_forward, return EE pos."""
        self.ik_data.qpos[self.q_ids] = q_arm
        mujoco.mj_forward(self.model, self.ik_data)
        return self.ik_data.site_xpos[self.site_id].copy()

    def __call__(self, obs):
        # ── 1. Sync isolated data with current env state ──────────────
        self.ik_data.qpos[:] = self.data.qpos[:]
        self.ik_data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.ik_data)
        q0_arm = self.ik_data.qpos[self.q_ids].copy()

        goal = obs["desired_goal"]

        # ── 2. Solve IK: minimise || FK(q) − goal ||² ────────────────
        result = sp_minimize(
            fun=lambda q: float(np.sum((self._fk(q) - goal) ** 2)),
            x0=q0_arm,
            method="L-BFGS-B",
            bounds=self.bounds,
            options={"maxiter": SCIPY_MAX_ITER, "ftol": 1e-7},
        )
        q_opt = result.x

        # ── 3. Restore ik_data to current state (clean for next call) ─
        self.ik_data.qpos[self.q_ids] = q0_arm
        mujoco.mj_forward(self.model, self.ik_data)

        # ── 4. Convert joint delta → EE velocity via Jacobian ─────────
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)

        dq_full            = np.zeros(self.model.nv)
        dq_full[self.v_ids] = q_opt - q0_arm
        dx_ee              = jacp @ dq_full

        # ── 5. Build action ───────────────────────────────────────────
        action = np.zeros(self.n_action)
        action[:3] = np.clip(dx_ee * self.gain, -1.0, 1.0)
        return action
