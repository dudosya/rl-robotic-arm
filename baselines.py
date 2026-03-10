# =============================================================================
# baselines.py — Classical (model-based) baseline policies for FetchPickAndPlace-v4.
#
# FetchPickAndPlace-v4 requires the arm to:
#   1. Move the gripper above the block object
#   2. Descend onto it
#   3. Grasp (close fingers)
#   4. Lift the block
#   5. Carry it to the goal position
#   6. Place (open fingers)
#
# This is fundamentally harder than FetchReach: classical methods must explicitly
# encode each phase, and grasping success is not guaranteed.
#
# Architecture
# ------------
# _BasePickPlacePolicy  —  shared state machine (HOVER → DESCEND → GRASP →
#                          LIFT → CARRY → PLACE).  Subclasses override
#                          _move_toward(obs, target) which computes the 3D
#                          velocity differently for each method.
#
# Subclasses
#   RandomPolicy         — random actions (no state machine)
#   PController          — proportional Cartesian velocity
#   JacobianPinvPolicy   — Jacobian pseudo-inverse IK velocity
#   ScipyIKPolicy        — numerical IK via L-BFGS-B per step
#
# All stateful policies expose  reset()  — call at the start of each episode.
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
# Observation helpers (FetchPickAndPlace-v4 layout)
# ─────────────────────────────────────────────────────────────────────────────
# obs["observation"] shape: (25,)
#   [0:3]   grip_pos        — gripper XYZ in world frame
#   [3:6]   object_pos      — block XYZ in world frame
#   [6:9]   object_rel_pos  — object_pos − grip_pos
#   [9:11]  gripper_state   — finger widths (both fingers)
#   [11:14] object_rot      — Euler angles of block
#   [14:17] object_velp     — block linear velocity
#   [17:20] object_velr     — block angular velocity
#   [20:23] grip_velp       — gripper linear velocity
#   [23:25] gripper_vel     — finger velocities
# obs["achieved_goal"]  = object XYZ  (NOT gripper!)
# obs["desired_goal"]   = target XYZ

def _grip_pos(obs):
    return obs["observation"][0:3].copy()

def _obj_pos(obs):
    return obs["observation"][3:6].copy()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Random Policy — no state machine
# ─────────────────────────────────────────────────────────────────────────────

class RandomPolicy:
    """Uniformly random actions — uninformed lower-bound baseline."""

    def __init__(self, env):
        self.action_space = env.action_space

    def reset(self):
        pass

    def __call__(self, obs):
        return self.action_space.sample()


# ─────────────────────────────────────────────────────────────────────────────
# Shared state-machine base class
# ─────────────────────────────────────────────────────────────────────────────

class _BasePickPlacePolicy:
    """State machine for pick-and-place.

    Phases
    ------
    0  HOVER_OVER_OBJ  — move EE to position above the block
    1  DESCEND_TO_OBJ  — lower EE to grasping height
    2  GRASP           — close gripper for GRASP_STEPS steps
    3  LIFT            — raise gripper (block should come with it)
    4  CARRY_TO_GOAL   — fly to goal XYZ with gripper closed
    5  PLACE           — open gripper at goal

    Subclasses must implement _move_toward(obs, target_pos) -> action (4,).
    The state machine sets action[3] (gripper command) automatically.
    """

    # Phase indices
    HOVER_OVER_OBJ = 0
    DESCEND_TO_OBJ = 1
    GRASP          = 2
    LIFT           = 3
    CARRY_TO_GOAL  = 4
    PLACE          = 5

    # Geometry constants
    HOVER_HEIGHT   = 0.06    # metres above block before descending
    LIFT_HEIGHT    = 0.15    # metres to lift block above its initial height
    XY_ALIGN_DIST  = 0.015   # xy distance threshold to start descending
    AT_TARGET_DIST = 0.025   # 3-D distance threshold: "arrived at waypoint"
    GRASP_STEPS    = 10      # steps spent closing gripper

    GRIPPER_OPEN   = -1.0
    GRIPPER_CLOSE  = +1.0

    def __init__(self, env):
        self.n_action = env.action_space.shape[0]
        self.reset()

    def reset(self):
        """Reset state machine — call at the start of every episode."""
        self.phase           = self.HOVER_OVER_OBJ
        self._grasp_counter  = 0
        self._initial_obj_z  = None   # set on first step to store table height

    def _move_toward(self, obs, target_pos) -> np.ndarray:
        """Return a 4-D action that steers the EE toward target_pos.
        Subclasses override this; state machine fills action[3] (gripper).
        """
        raise NotImplementedError

    @staticmethod
    def _dist3(a, b):
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

    def __call__(self, obs):
        grip = _grip_pos(obs)
        obj  = _obj_pos(obs)
        goal = obs["desired_goal"].copy()

        # Capture object height on first step of the episode
        if self._initial_obj_z is None:
            self._initial_obj_z = float(obj[2])

        action = np.zeros(self.n_action)

        # ── Phase 0: hover above block ────────────────────────────────
        if self.phase == self.HOVER_OVER_OBJ:
            target = obj + np.array([0.0, 0.0, self.HOVER_HEIGHT])
            action = self._move_toward(obs, target)
            action[3] = self.GRIPPER_OPEN
            if self._dist3(grip, target) < self.AT_TARGET_DIST:
                self.phase = self.DESCEND_TO_OBJ

        # ── Phase 1: descend onto block ───────────────────────────────
        elif self.phase == self.DESCEND_TO_OBJ:
            target = obj + np.array([0.0, 0.0, 0.005])
            action = self._move_toward(obs, target)
            action[3] = self.GRIPPER_OPEN
            # Advance when XY is aligned (don't wait for perfect Z)
            if self._dist3(grip[:2], obj[:2]) < self.XY_ALIGN_DIST:
                self.phase = self.GRASP
                self._grasp_counter = 0

        # ── Phase 2: close gripper ────────────────────────────────────
        elif self.phase == self.GRASP:
            action[:3] = 0.0          # hold position
            action[3]  = self.GRIPPER_CLOSE
            self._grasp_counter += 1
            if self._grasp_counter >= self.GRASP_STEPS:
                self.phase = self.LIFT

        # ── Phase 3: lift ─────────────────────────────────────────────
        elif self.phase == self.LIFT:
            lift_target = np.array([
                obj[0], obj[1],
                self._initial_obj_z + self.LIFT_HEIGHT
            ])
            action = self._move_toward(obs, lift_target)
            action[3] = self.GRIPPER_CLOSE
            # Advance when block is sufficiently lifted
            if obj[2] > self._initial_obj_z + self.LIFT_HEIGHT - 0.03:
                self.phase = self.CARRY_TO_GOAL

        # ── Phase 4: carry to goal ────────────────────────────────────
        elif self.phase == self.CARRY_TO_GOAL:
            action = self._move_toward(obs, goal)
            action[3] = self.GRIPPER_CLOSE
            if self._dist3(obj, goal) < self.AT_TARGET_DIST * 1.5:
                self.phase = self.PLACE

        # ── Phase 5: place (open gripper) ─────────────────────────────
        elif self.phase == self.PLACE:
            action[:3] = 0.0
            action[3]  = self.GRIPPER_OPEN

        return action


# ─────────────────────────────────────────────────────────────────────────────
# Shared MuJoCo helpers (for Jacobian / Scipy policies)
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
# 2. Proportional Cartesian Controller
# ─────────────────────────────────────────────────────────────────────────────

class PController(_BasePickPlacePolicy):
    """Proportional (P) controller in Cartesian space.

    _move_toward simply scales the positional error by a fixed gain K_p:

        action[:3] = clip(K_p * (target - grip_pos), -1, 1)

    No kinematic model is used — only the Cartesian positions from the
    observation dictionary.
    """

    def __init__(self, env, gain: float = P_GAIN):
        super().__init__(env)
        self.gain = gain

    def _move_toward(self, obs, target_pos) -> np.ndarray:
        grip  = _grip_pos(obs)
        dx    = target_pos - grip
        action = np.zeros(self.n_action)
        action[:3] = np.clip(self.gain * dx, -1.0, 1.0)
        return action


# ─────────────────────────────────────────────────────────────────────────────
# 3. Jacobian Pseudo-Inverse Controller
# ─────────────────────────────────────────────────────────────────────────────

class JacobianPinvPolicy(_BasePickPlacePolicy):
    """Jacobian pseudo-inverse controller (differential / first-order IK).

    At each step:
      1. Compute the translational Jacobian J ∈ R^{3 × nv} via mj_jacSite().
      2. Compute positional error  Δx = target − grip_pos.
      3. Compute joint delta:       Δq = J⁺ Δx   (Moore-Penrose pseudo-inverse)
      4. Recover EE velocity:       Δx_ee = J Δq  (= Δx when J has full row rank)
      5. Clip and scale into the action range.

    This is the standard model-based IK approach from the project description.
    """

    def __init__(self, env, gain: float = P_GAIN):
        super().__init__(env)
        self.model   = env.unwrapped.model
        self.data    = env.unwrapped.data
        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE
        )
        self.gain = gain

    def _move_toward(self, obs, target_pos) -> np.ndarray:
        grip = _grip_pos(obs)
        dx   = target_pos - grip                          # (3,)

        # Translational Jacobian J ∈ R^{3 × nv}
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        J = jacp

        # Δq = J⁺ Δx  →  Δx_ee = J Δq
        J_pinv = np.linalg.pinv(J)
        dq     = J_pinv @ dx
        dx_ee  = J @ dq

        action = np.zeros(self.n_action)
        action[:3] = np.clip(dx_ee * self.gain, -1.0, 1.0)
        return action


# ─────────────────────────────────────────────────────────────────────────────
# 4. Scipy Numerical IK Policy
# ─────────────────────────────────────────────────────────────────────────────

class ScipyIKPolicy(_BasePickPlacePolicy):
    """Numerical IK controller using scipy L-BFGS-B optimisation.

    Formulates a global joint-space optimisation at every step:

        q* = argmin_q  || FK(q) − target ||²

    where FK(q) returns the gripper site position for joint configuration q.
    An isolated MjData copy is used so the live environment is never altered.
    The optimal joint delta Δq = q* − q₀ is mapped back to a Cartesian action
    via the Jacobian: Δx_ee = J Δq.
    """

    def __init__(self, env, gain: float = P_GAIN):
        super().__init__(env)
        self.model   = env.unwrapped.model
        self.data    = env.unwrapped.data
        self.ik_data = mujoco.MjData(self.model)    # isolated copy for FK queries
        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE
        )
        self.q_ids, self.v_ids = _get_arm_indices(self.model)
        self.bounds  = _get_joint_bounds(self.model, self.q_ids)
        self.gain    = gain

    def _fk(self, q_arm) -> np.ndarray:
        """Forward kinematics using the isolated data copy."""
        self.ik_data.qpos[self.q_ids] = q_arm
        mujoco.mj_forward(self.model, self.ik_data)
        return self.ik_data.site_xpos[self.site_id].copy()

    def _move_toward(self, obs, target_pos) -> np.ndarray:
        # Sync isolated copy with current env state
        self.ik_data.qpos[:] = self.data.qpos[:]
        self.ik_data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.ik_data)
        q0_arm = self.ik_data.qpos[self.q_ids].copy()

        # Solve IK
        result = sp_minimize(
            fun=lambda q: float(np.sum((self._fk(q) - target_pos) ** 2)),
            x0=q0_arm,
            method="L-BFGS-B",
            bounds=self.bounds,
            options={"maxiter": SCIPY_MAX_ITER, "ftol": 1e-7},
        )
        q_opt = result.x

        # Restore isolated copy
        self.ik_data.qpos[self.q_ids] = q0_arm
        mujoco.mj_forward(self.model, self.ik_data)

        # Convert joint delta → Cartesian velocity via Jacobian
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)

        dq_full             = np.zeros(self.model.nv)
        dq_full[self.v_ids] = q_opt - q0_arm
        dx_ee               = jacp @ dq_full

        action = np.zeros(self.n_action)
        action[:3] = np.clip(dx_ee * self.gain, -1.0, 1.0)
        return action
