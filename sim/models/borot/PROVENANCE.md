# Vendored model

- Source: https://github.com/Seeed-Projects/Borot-Arm_Mujoco
- Commit: `3f3e98d0682fcc382d5607f92890f44918d097d9` (2026-08-28)
- File: `reBotArm_ros2_DM/src/rebotarm_mujoco/models/rebotarm_b601_colored.xml` — copied **unmodified**.
- Meshes: only the files referenced by that MJCF, copied from
  `reBotArm_ros2_DM/src/rebotarm_bringup/description/meshes/` (the MJCF's own `meshdir="../meshes"` does
  not exist upstream; `sim/scene.py` points `meshdir` at `./meshes` when loading).
- License: CERN-OHL-W-2.0 for the hardware/model files (see `LICENSE`, copied from the repository root).

All changes (floor, block, link collision hulls, finger coupling, actuators, colours, excluded hull
pairs) are applied in memory by `sim/scene.py`; this directory is not edited.

Checked against the control SDK: at the real arm's joint angles, MuJoCo's `end_link` equals the
SDK's Pinocchio FK (`reBot-DevArm_fixend.urdf`) to < 1e-6 mm / 0 deg (`sim/verify_sim.py`).
Related upstream repos checked on 2026-10-13: Seeed-Projects/reBot-DevArm `9ac6169`,
Seeed-Projects/reBotArm_control_py `6415d43` (local copy in rebot/ is based on `40ab6ce`).
