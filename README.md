# Reinforcement Learning vs. Classical Kinematics for Robotic Arm Control

**Course Final Project**

This project investigates whether a **Reinforcement Learning** agent can match or
exceed classical **model-based** controllers on a 3-D robotic reaching task � without
any explicit knowledge of the robot's kinematic model.

---

## Environment

**FetchPickAndPlace-v4** (MuJoCo via `gymnasium-robotics`)

The 7-DOF Fetch Research robot arm must move its gripper end-effector to a randomly
sampled 3-D target position in space.

- **Observation:** dict with `observation` (25-D joint state), `achieved_goal` (EE position), `desired_goal` (target position)
- **Action:** 4-D Cartesian velocity `[dx, dy, dz, gripper]` clipped to `[-1, 1]`
- **Reward:** dense � `-||EE - goal||` (continuous gradient signal)
- **Episode length:** 50 steps

---

## Methods Compared

| #   | Method                  | Type           | Description                                                     |
| --- | ----------------------- | -------------- | --------------------------------------------------------------- | --- | ------------ | --- | --- |
| 1   | Random Policy           | Baseline       | Uniformly random actions � lower bound                          |
| 2   | P-Controller            | Classical      | Velocity proportional to Cartesian error; no joint model needed |
| 3   | Jacobian Pseudo-Inverse | Classical (IK) | delta_q = J+ delta_x � standard differential kinematics         |
| 4   | Scipy Numerical IK      | Classical (IK) | Global optimisation: argmin_q                                   |     | FK(q) - goal |     | ^2  |
| 5   | SAC (trained)           | RL             | Learns policy from scratch via environment interaction          |

---

## Project Structure

```
rl-robotic-arm/
+-- main.py            <- master runner � executes the full experiment top to bottom
+-- config.py          <- all hyperparameters and constants (edit here to tune)
+-- baselines.py       <- classical policy implementations
+-- train_sac.py       <- SAC model building and training loop
+-- evaluate.py        <- shared evaluation function + plotting utilities
+-- videos.py          <- demo episode recording and video saving
+-- requirements.txt
L-- README.md
```

---

## How to Run on Google Colab

### Step 1 � Clone the repository

```
!git clone https://github.com/<your-username>/rl-robotic-arm.git
%cd rl-robotic-arm
```

### Step 2 � Enable GPU

Go to **Runtime > Change runtime type > T4 GPU** before running.

### Step 3 � Install dependencies

```
!pip install -r requirements.txt
```

After installation, go to **Runtime > Restart runtime**, then continue from Step 4.

### Step 4 � Run the full experiment

```
!python main.py
```

That is it. The script runs everything sequentially and prints progress to the console.

---

## Running Locally (Windows / Mac / Linux)

The project runs fine on a local machine — the SAC model is tiny (~250K parameters),
so no GPU is required (CPU training just takes longer).

```bash
git clone https://github.com/<your-username>/rl-robotic-arm.git
cd rl-robotic-arm

# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python main.py
```

**Windows note:** `pyvirtualdisplay` (Linux X11 only) is automatically skipped on
Windows — `main.py` detects the OS and skips it. MuJoCo renders natively on Windows
so no extra setup is needed.

**Estimated local runtime (CPU only):**
| Phase | Time |
|-------|------|
| Classical baselines | ~10–15 min |
| SAC training (500k steps, CPU) | ~3–5 hours |
| Plots + videos | ~5 min |

> For CPU runs, consider reducing `TOTAL_TIMESTEPS` to `150_000` in `config.py`
> to bring training down to ~1–1.5 hours. Results will be less polished but
> the comparison still holds.

---

## Estimated Runtime (Colab T4 GPU)

| Phase                     | Time                             |
| ------------------------- | -------------------------------- |
| Classical baselines (x4)  | ~8 min (Scipy IK is the slowest) |
| SAC training (150k steps) | ~55-75 min                       |
| Plots + videos            | ~3 min                           |
| **Total**                 | **~65-85 min**                   |

---

## Output Files

All outputs are written to `./outputs/`:

```
outputs/
+-- plots/
�   +-- training_curve.png      <- SAC reward over training steps + baseline overlays
�   L-- comparison_chart.png    <- side-by-side bar chart (mean reward + success rate)
+-- videos/
�   +-- demo_random.mp4
�   +-- demo_p_controller.mp4
�   +-- demo_jacobian_ik.mp4
�   +-- demo_scipy_ik.mp4
�   L-- demo_sac_trained.mp4
+-- models/
�   L-- best_model.zip          <- best SAC checkpoint (saved by EvalCallback)
L-- logs/
    +-- evaluations.npz         <- per-checkpoint eval data (used for training curve)
    L-- sac_tensorboard/        <- TensorBoard logs
```

---

## Re-running Without Retraining

After the first run, you can regenerate plots and videos without retraining by
setting `SKIP_TRAINING = True` in `config.py` and running again:

```
# In config.py:
SKIP_TRAINING = True

# Then:
!python main.py
```

---

## Tuning Hyperparameters

All hyperparameters live in `config.py`. Key ones to try:

| Parameter         | Default | Effect                                                              |
| ----------------- | ------- | ------------------------------------------------------------------- |
| `TOTAL_TIMESTEPS` | 150,000 | More steps = better SAC performance                                 |
| `P_GAIN`          | 10.0    | Higher gain = faster but potentially unstable classical controllers |
| `SCIPY_MAX_ITER`  | 25      | More iterations = better IK accuracy, slower per step               |
| `SEED`            | 42      | Change for different random seeds                                   |

---

## Key Findings (Expected)

- **P-Controller and Jacobian IK** achieve high success rates because the action space is already Cartesian.
- **Scipy IK** is the most accurate classical method but slowest at inference.
- **SAC** achieves competitive success rates after ~150k steps _without any kinematic model_.
- The key RL advantage: **adaptability** � if the robot geometry changes, retrain; classical methods require re-deriving the kinematic model.

---

## Dependencies

- [gymnasium-robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics) � FetchPickAndPlace-v4 environment
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) � SAC implementation
- [MuJoCo](https://mujoco.org/) � physics simulation (installed automatically)
- [scipy](https://scipy.org/) � numerical IK optimisation
- [imageio](https://imageio.readthedocs.io/) � video recording
- [pyvirtualdisplay](https://github.com/ponty/PyVirtualDisplay) � headless rendering on Colab
