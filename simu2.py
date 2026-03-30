import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import subprocess

# ==========================================
# 1. Config (目标值拆分为标量)
# ==========================================
RUN_MODE = 'PLOT'       
CTRL_MODE = 'BALANCE'    
SIM_DURATION = 120.0     # 延长到120秒测试稳定性

# X 方向 (纵向 - 轮子控制)
TARGET_X = 2      
TARGET_VX = 0.0     

# Y 方向 (横向 - 滑块控制)
TARGET_Y = 0      
TARGET_VY = 0.0     

# ==========================================
# 2. Parameters (针对永不倒下的强固增益)
# ==========================================
class Params:
    # --- 纵向控制 (X / Main Motor) ---
    K_pitch = 30.0    # 增强纵向刚度
    K_dpitch = 5.0
    K_pos_x = 2.0     
    K_vel_x = 4.0     
    
    # --- 横向控制 (Y / Slider Motor) ---
    # 核心：K_roll必须远大于K_s_res，确保救车力矩永远优先于复位力矩
    K_roll = 350.0    # 极高增益，对抗大角度倾斜
    K_droll = 180.0   # 高阻尼，吸收滑块撞墙产生的冲击波
    K_s_res = 12.0    # 极低复位，只要能缓慢回中即可，绝不干扰平衡
    K_ds_res = 15.0   # 配合物理阻尼缺失
    
    K_pos_y = 0.0   
    K_vel_y = 0.0   
    
    max_tilt = (150/180)*np.pi

# ==========================================
# 3. State Extraction (Abs Coordinate)
# ==========================================
def get_abs_state(data):
    root = data.joint('root')
    box_hinge = data.joint('box_hinge')
    slider = data.joint('slider_joint')

    # 位置与速度
    x_c, y_c = root.qpos[0], root.qpos[1]
    vx, vy = root.qvel[0], root.qvel[1]

    body_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, 'wheel')
    mat = data.xmat[body_id].reshape(3, 3)

    # wheel 的局部 y 轴 = 轮轴方向
    wheel_axis = mat[:, 1]
    roll = np.arctan2(wheel_axis[2], np.sqrt(wheel_axis[0]**2 + wheel_axis[1]**2))

    # pitch 看 box_link
    body_id_box = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, 'box_link')
    mat_box = data.xmat[body_id_box].reshape(3, 3)
    z_axis_box = mat_box[:, 2]
    pitch = np.arctan2(z_axis_box[0], np.sqrt(z_axis_box[1]**2 + z_axis_box[2]**2))

    # 角速度
    roll_dot = root.qvel[3]
    pitch_dot = root.qvel[4]
    abs_gamma_dot = pitch_dot + box_hinge.qvel[0]

    return {
        'pos': np.array([x_c, y_c]),
        'vel': np.array([vx, vy]),
        'pitch': pitch,
        'pitch_dot': abs_gamma_dot,
        'roll': roll,
        'roll_dot': roll_dot,
        'slider_pos': slider.qpos[0],
        'slider_vel': slider.qvel[0]
    }

# ==========================================
# 4. Controller (逻辑保持不变)
# ==========================================
class SegwayController3D:
    def __init__(self):
        self.tx, self.ty = TARGET_X, TARGET_Y
        self.tvx, self.tvy = TARGET_VX, TARGET_VY

    def compute(self, x, y, vx, vy, pitch, pitch_dot, roll, roll_dot, slider_pos, slider_vel):
        # ===== 1. 纵向控制 =====
        if CTRL_MODE == 'POSITION':
            tau_pitch = (Params.K_pitch * (0 - pitch) + 
                         Params.K_dpitch * (0 - pitch_dot) + 
                         Params.K_pos_x * (self.tx - x) + 
                         Params.K_vel_x * (0 - vx))
        elif CTRL_MODE == 'VELOCITY':
            tau_pitch = (Params.K_pitch * (0 - pitch) + 
                         Params.K_dpitch * (0 - pitch_dot) + 
                         Params.K_vel_x * (self.tvx - vx))
        else: # BALANCE
            tau_pitch = Params.K_pitch * (0 - pitch) + Params.K_dpitch * (0 - pitch_dot)

        # ===== 2. 横向控制 =====
        roll_term = Params.K_roll * roll + Params.K_droll * roll_dot

        if CTRL_MODE == 'POSITION':
            slider_force = (roll_term + Params.K_pos_y * (self.ty - y) + Params.K_vel_y * (0 - vy))
        elif CTRL_MODE == 'VELOCITY':
            slider_force = (roll_term + Params.K_vel_y * (self.tvy - vy))
        else: # BALANCE
            # 增益压制逻辑：roll_term 的优先级由于参数设定远高于复位项
            slider_force = roll_term - Params.K_s_res * slider_pos - Params.K_ds_res * slider_vel

        return -tau_pitch, slider_force

# ==========================================
# 6. Simulation
# ==========================================
def main():
    model = mujoco.MjModel.from_xml_path('Model/out.xml') 
    data = mujoco.MjData(model)

    pitch_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'main_motor')
    slider_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'slider_motor')

    controller = SegwayController3D()

    history = {'t': [], 'x': [], 'x_dot': [], 'gamma': [], 'roll': [],'slider_pos':[]}

    if RUN_MODE == 'VIEWER':
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time < SIM_DURATION:
                step_start = time.time()
                state = get_abs_state(data)
                torque_p, force_s = controller.compute(
                    state['pos'][0], state['pos'][1], state['vel'][0], state['vel'][1],
                    state['pitch'], state['pitch_dot'], state['roll'], state['roll_dot'],
                    state['slider_pos'], state['slider_vel']
                )
                data.ctrl[pitch_motor_id] = np.clip(torque_p, -150, 150)
                data.ctrl[slider_motor_id] = np.clip(force_s, -100, 100)
                mujoco.mj_step(model, data)
                viewer.sync()
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    elif RUN_MODE == 'PLOT':
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        print(f"Simulating for {CTRL_MODE} (Long duration stability test) ...")
        while data.time < SIM_DURATION:
            state = get_abs_state(data)
            torque_p, force_s = controller.compute(
                state['pos'][0], state['pos'][1], state['vel'][0], state['vel'][1],
                state['pitch'], state['pitch_dot'], state['roll'], state['roll_dot'],
                state['slider_pos'], state['slider_vel']
            )
            data.ctrl[pitch_motor_id] = np.clip(torque_p, -150, 150)
            data.ctrl[slider_motor_id] = np.clip(force_s, -100, 100)
            mujoco.mj_step(model, data)
            
            history['t'].append(data.time)
            history['x'].append(state['pos'][0])
            history['gamma'].append(state['pitch'])
            history['roll'].append(state['roll'])
            history['slider_pos'].append(state['slider_pos'])

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
        ax1.plot(history['t'], history['x'], color='orange', label='Actual $x_c$')
        ax1.set_title(f"Stability Analysis - 120s Run"); ax1.grid(True); ax1.legend()
        
        ax2.plot(history['t'], np.degrees(history['gamma']), color='green', label='Pitch')
        ax2.set_ylabel("Pitch (deg)"); ax2.grid(True); ax2.legend()

        ax3.plot(history['t'], np.degrees(history['roll']), color='blue', label='Roll')
        ax3.set_ylabel("Roll (deg)"); ax3.grid(True); ax3.legend()

        ax4.plot(history['t'], history['slider_pos'], color='red', label='Slider Pos')
        ax4.set_ylabel("Slider (m)"); ax4.set_xlabel("Time (s)"); ax4.grid(True); ax4.legend()
        ax4.set_ylim([-0.11, 0.11]) # 观察是否撞墙

        plt.tight_layout()
        plt.savefig('long_term_stable.png')
        print("仿真完成。结果已保存为 long_term_stable.png")

if __name__ == "__main__":
    main()