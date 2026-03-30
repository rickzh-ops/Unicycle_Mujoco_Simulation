import inspect
import os
import traceback

import mujoco
import numpy as np
import optuna

import simu


# =========================================================
# Helpers
# =========================================================
def quat_from_roll(roll_rad: float) -> np.ndarray:
    """Quaternion [w, x, y, z] for pure roll about x-axis."""
    half = 0.5 * roll_rad
    q = np.array([np.cos(half), np.sin(half), 0.0, 0.0], dtype=float)
    q /= np.linalg.norm(q)
    return q


def safe_build_controller(model, slider_motor_id):
    """
    Try several constructor signatures to support your different simu.py versions.
    """
    ctor = simu.SegwayController3D
    try:
        sig = inspect.signature(ctor)
        n_params = len(sig.parameters)
    except Exception:
        n_params = None

    # Try common variants
    try:
        return ctor()
    except TypeError:
        pass

    try:
        return ctor(model, slider_motor_id)
    except TypeError:
        pass

    try:
        return ctor(model)
    except TypeError:
        pass

    # Last resort: let it fail loudly
    return ctor()


def get_actuator_ctrl_limit(model, actuator_name: str, default_limit: float) -> float:
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    if act_id < 0:
        return default_limit
    if model.actuator_ctrllimited[act_id]:
        lo, hi = model.actuator_ctrlrange[act_id]
        return float(max(abs(lo), abs(hi)))
    return default_limit


def get_joint_pos_limit(model, joint_name: str, default_limit: float) -> float:
    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if j_id < 0:
        return default_limit
    if model.jnt_limited[j_id]:
        lo, hi = model.jnt_range[j_id]
        return float(max(abs(lo), abs(hi)))
    return default_limit


def initialize_state(model, data, roll0_rad: float, slider0: float = 0.0):
    """
    Initialize upright-ish state with a small roll perturbation.
    Assumes root is a freejoint.
    """
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    if data.act is not None and data.act.size > 0:
        data.act[:] = 0.0

    # freejoint qpos layout: [x, y, z, qw, qx, qy, qz]
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = 0.25

    q = quat_from_roll(roll0_rad)
    data.qpos[3:7] = q

    # keep pitch hinge at zero
    try:
        data.joint('box_hinge').qpos[0] = 0.0
        data.joint('box_hinge').qvel[0] = 0.0
    except Exception:
        pass

    try:
        data.joint('slider_joint').qpos[0] = slider0
        data.joint('slider_joint').qvel[0] = 0.0
    except Exception:
        pass

    mujoco.mj_forward(model, data)


# =========================================================
# Objective
# =========================================================
def objective(trial: optuna.Trial) -> float:
    # -----------------------------------------------------
    # 1) Search space
    # -----------------------------------------------------
    # Keep sign freedom, but shrink from your original extremely large ranges.
    simu.Params.K_roll = trial.suggest_float("K_roll", -800.0, 800.0)
    simu.Params.K_droll = trial.suggest_float("K_droll", -800.0, 800.0)

    simu.Params.K_s_res = trial.suggest_float("K_s_res", -100.0, 100.0)
    simu.Params.K_ds_res = trial.suggest_float("K_ds_res", -100.0, 100.0)

    # lock longitudinal loop as you requested
    simu.Params.K_pitch = 3.0
    simu.Params.K_dpitch = 0.8

    # -----------------------------------------------------
    # 2) Load model
    # -----------------------------------------------------
    xml_path = os.path.join("Model", "out.xml") if os.path.exists("Model") else "out.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    p_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "main_motor")
    s_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "slider_motor")

    controller = safe_build_controller(model, s_id)

    # read actual limits from XML/model
    slider_ctrl_limit = get_actuator_ctrl_limit(model, "slider_motor", 100.0)
    slider_pos_limit = get_joint_pos_limit(model, "slider_joint", 0.1)

    # -----------------------------------------------------
    # 3) Initial perturbation
    # -----------------------------------------------------
    roll0 = np.radians(3.0)
    initialize_state(model, data, roll0_rad=roll0, slider0=0.0)

    # -----------------------------------------------------
    # 4) Simulation / cost
    # -----------------------------------------------------
    total_cost = 0.0
    eval_duration = 60.0

    # small reward for survival time through lower cumulative cost
    while data.time < eval_duration:
        state = simu.get_abs_state(data)

        torque_p, force_s = controller.compute(
            state["pos"][0], state["pos"][1],
            state["vel"][0], state["vel"][1],
            state["pitch"], state["pitch_dot"],
            state["roll"], state["roll_dot"],
            state["slider_pos"], state["slider_vel"]
        )

        data.ctrl[p_id] = np.clip(torque_p, -150.0, 150.0)
        data.ctrl[s_id] = np.clip(force_s, -slider_ctrl_limit, slider_ctrl_limit)

        try:
            mujoco.mj_step(model, data)
        except Exception:
            return 1e20

        # ------------- hard failure checks -------------
        curr_roll_deg = abs(np.degrees(state["roll"]))

        curr_slider = abs(state["slider_pos"])
        if abs(state["roll"]) > np.radians(40):
            return 1e14 + 1e10 * (eval_duration - data.time)

        if curr_roll_deg > 20.0 or curr_slider > 0.08:
            return 1e12 + 1e9 * (eval_duration - data.time)

        if curr_slider > 0.95 * slider_pos_limit:
            return 5e11 + 1e9 * (eval_duration - data.time)

        # ------------- running cost -------------
        # Strongly prioritize keeping roll small
        # Then suppress velocity, slider motion, and control chatter
        total_cost += (
            10000.0 * (state["roll"] ** 2)
            + 500.0  * (state["roll_dot"] ** 2)
            + 200.0  * (state["slider_pos"] ** 2)
        )

        # Extra penalty if nearing large tilt
        if curr_roll_deg > 15.0:
            total_cost += 2000.0 * ((curr_roll_deg - 15.0) / 20.0) ** 2

    return total_cost


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    print(">>> 正在启动【PD 全域符号 + 合理边界】调参...")

    try:
        study.optimize(objective, n_trials=2000, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\n调参被手动中断，输出当前最好结果。")
    except Exception:
        print("\n调参过程中出错：")
        traceback.print_exc()

    print("\n" + "=" * 50)
    print("【当前最佳参数】请更新 simu.py：")
    if study.best_trial is not None:
        best = study.best_params
        for k, v in best.items():
            print(f"    {k} = {v:.6f}")
        print(f"\nBest objective = {study.best_value:.6e}")
    else:
        print("没有得到有效结果。")
    print("=" * 50)