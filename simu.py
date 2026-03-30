import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import subprocess

# ==========================================
# 1. Config (目标值拆分为标量)
# ==========================================
RUN_MODE = 'VIEWER'       
CTRL_MODE = 'BALANCE'    
SIM_DURATION = 100.0     

# X 方向 (纵向 - 轮子控制)
TARGET_X = 0     
TARGET_VX = 0.0     

# Y 方向 (横向 - 滑块控制)
TARGET_Y = 0      
TARGET_VY = 0.0     

# ==========================================
# 2. Parameters (增益拆分为标量)
# ==========================================
class Params:
    # --- 纵向控制 (X / Main Motor) ---
    K_pitch  = 3
    K_dpitch = 0.8
    K_pos_x = 2     #2
    K_vel_x = 4     #4
    
    # --- 横向控制 (Y / Linear Motor) ---
    #Balance:
    K_roll = 140
    K_droll = 120
    K_s_res = 80
    K_ds_res = 67.5
    # K_roll = 0
    # K_droll = 0
    # K_s_res = 0
    # K_ds_res = 0
    K_pos_y = 0.0   
    K_vel_y = 0.0   

    max_tilt = (150/180)*np.pi

# ==========================================
# 3. State Extraction (Abs Coordinate)
# ==========================================
# def get_abs_state(data):
#     # Abs Angle
#     body_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, 'box_link')
#     mat = data.xmat[body_id].reshape(3, 3)
#     z_axis = mat[:, 2] 
#     abs_gamma = np.arctan2(z_axis[0], z_axis[2])

#     # 2-D state Extraction (only contains pitch)
#     x_c = data.qpos[0]
#     x_c_dot = data.qvel[0]
    
#     # Abs omega
#     # wheel_pitch_dot = data.joint('root_y').qvel[0]
#     # hinge_pitch_dot = data.joint('box_hinge').qvel[0]
#     # abs_gamma_dot = wheel_pitch_dot + hinge_pitch_dot
#     wheel_pitch_dot = data.joint('root_y').qvel[0]
#     hinge_pitch_dot = data.joint('box_hinge').qvel[0]
#     abs_gamma_dot = wheel_pitch_dot + hinge_pitch_dot
    
#     return [x_c, x_c_dot, abs_gamma, abs_gamma_dot]

def get_abs_state(data):
    root = data.joint('root')
    box_hinge = data.joint('box_hinge')
    slider = data.joint('slider_joint')

    # 位置与速度
    x_c, y_c = root.qpos[0], root.qpos[1]
    vx, vy = root.qvel[0], root.qvel[1]

    # 获取旋转矩阵
    # body_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, 'wheel')
    # mat = data.xmat[body_id].reshape(3, 3)
    # z_axis = mat[:, 2] 

    # # 鲁棒的角度提取：加入极小值防止分母为0
    # eps = 1e-8
    # # Pitch (纵向)
    # pitch = np.arctan2(z_axis[0], z_axis[2] + eps)
    # # Roll (横向)
    # roll = np.arctan2(-z_axis[1], z_axis[2] + eps) 
    body_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, 'wheel')
    mat = data.xmat[body_id].reshape(3, 3)

    # wheel 的局部 y 轴 = 轮轴方向
    wheel_axis = mat[:, 1]

    # signed roll
    roll = np.arctan2(wheel_axis[2], wheel_axis[1])

    # pitch 继续单独看 box_link
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
# 4. Controller
# ==========================================
# class SegwayController:
#     def __init__(self):
#         self.last_t = 0.0

#     def compute(self, state, t):
#         x, x_dot, gamma, gamma_dot = state
#         dt = t - self.last_t if t > self.last_t else 0.0005
#         self.last_t = t

#         if CTRL_MODE == 'POSITION':
#             pos_err = TARGET_POS - x
#             tau =Params.K_gamma*(0-gamma) + Params.K_dgamma*(0-gamma_dot) + Params.K_position * pos_err + Params.K_velocity * (0 - x_dot)
#             #tau = np.clip(tau, -Params.max_tau, Params.max_tau)

#         elif CTRL_MODE == 'VELOCITY':
#             vel_err = TARGET_VEL - x_dot
#             tau = Params.K_velocity * vel_err  + Params.K_gamma*(0-gamma) + Params.K_dgamma*(0-gamma_dot) + Params.K_i*(vel_err*dt)
#             #tau = np.clip(tau, -Params.max_tau, Params.max_tau)

#         else: # BALANCE
#             tau = Params.K_gamma*(0-gamma) + Params.K_dgamma*(0-gamma_dot)
#             #tau = np.clip(tau, -Params.max_tau, Params.max_tau)

#         return -tau

class SegwayController3D:
    def __init__(self):
        self.tx, self.ty = TARGET_X, TARGET_Y
        self.tvx, self.tvy = TARGET_VX, TARGET_VY

    def compute(self, x, y, vx, vy, pitch, pitch_dot, roll, roll_dot, slider_pos, slider_vel):
        # ===== 1. 纵向控制 (轮子电机 -> 负责 X 轴) =====
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

        # ===== 2. 横向控制 (滑块电机 -> 负责 Y 轴) =====
        # if abs(roll) < np.radians(20):
        #     roll_eff = roll
        # else:
        #     roll_eff = np.sign(roll) * np.radians(20)
        roll_term = Params.K_roll * roll + Params.K_droll * roll_dot

        if CTRL_MODE == 'POSITION':
            slider_force = (roll_term + 
                            Params.K_pos_y * (self.ty - y) + 
                            Params.K_vel_y * (0 - vy))
            
        elif CTRL_MODE == 'VELOCITY':
            slider_force = (roll_term + 
                            Params.K_vel_y * (self.tvy - vy))
            
        else: # BALANCE
            # 基础自稳 + 滑块中心复位 + 阻尼
            slider_force = (
                Params.K_roll * roll
                + Params.K_droll * roll_dot
                - Params.K_s_res * slider_pos
                - Params.K_ds_res * slider_vel
            )

        # 为了防止瞬间冲击导致飞出
        slider_force = np.clip(slider_force, -100, 100)

        return -tau_pitch, slider_force
# ==========================================
# 6. Simulation
# ==========================================
# def main():
#     model = mujoco.MjModel.from_xml_path('Model/out.xml') 


#     data = mujoco.MjData(model)

#     # Debug
#     body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'wheel')
#     print(f"Wheel_Mass: {model.body_mass[body_id]}")
#     print(f"Wheel_Inertia (diagonal): {model.body_inertia[body_id]}")

#     body_id2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'box_link')
#     print(f"Box_Mass: {model.body_mass[body_id2]}")
#     print(f"Box_Inertia (diagonal): {model.body_inertia[body_id2]}")


#     controller = SegwayController3D()
#     data.joint('box_hinge').qpos[0] = 0.0 # Perturbation

#     history = {'t': [], 'x': [], 'x_dot': [], 'gamma': []}
def main():
    model = mujoco.MjModel.from_xml_path('Model/out.xml') 
    data = mujoco.MjData(model)

    # 获取执行器 ID
    pitch_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'main_motor')
    slider_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'slider_motor')

    controller = SegwayController3D()
    #data.joint('box_hinge').qpos[0] = np.radians(2.0) 

    # 绘图数据记录
    history = {'t': [], 'x': [], 'x_dot': [], 'gamma': [], 'roll': [],'slider_pos':[]}

    if RUN_MODE == 'VIEWER':
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time < SIM_DURATION:
                step_start = time.time()
                
                state = get_abs_state(data)
                torque_p, force_s = controller.compute(
                    x         = state['pos'][0],
                    y         = state['pos'][1],
                    vx        = state['vel'][0],
                    vy        = state['vel'][1],
                    pitch     = state['pitch'],
                    pitch_dot = state['pitch_dot'],
                    roll      = state['roll'],
                    roll_dot  = state['roll_dot'],
                    slider_pos= state['slider_pos'],
                    slider_vel = state['slider_vel']
                )
                
                #torque_p, force_s = controller.compute(state, data.time)
                
                # 控制量输出
                data.ctrl[pitch_motor_id] = np.clip(torque_p, -150, 150)
                data.ctrl[slider_motor_id] = np.clip(force_s, -100, 100)
                
                #Debug
                # if data.time < 4.0: 
                #     print(f"T: {data.time:.3f} | Roll: {state['roll']:.3f} | Force: {force_s:.2f} | Slid_Pos: {state['slider_pos']:.3f}")
                
                mujoco.mj_step(model, data)
                viewer.sync()
                
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    elif RUN_MODE == 'PLOT':
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import pandas as pd
        print(f"Simulating for {CTRL_MODE} (Non-stop mode) ...")
        while data.time < SIM_DURATION:
            state = get_abs_state(data)
            torque_p, force_s = controller.compute(
                state['pos'][0], state['pos'][1],
                state['vel'][0], state['vel'][1],
                state['pitch'], state['pitch_dot'],
                state['roll'], state['roll_dot'],
                state['slider_pos'], state['slider_vel']
            )
            
            data.ctrl[pitch_motor_id] = np.clip(torque_p, -150, 150)
            data.ctrl[slider_motor_id] = np.clip(force_s, -20, 20)
            
            mujoco.mj_step(model, data)
            
            # 记录数据用于后续画图
            history['t'].append(data.time)
            history['x'].append(state['pos'][0])
            history['x_dot'].append(state['vel'][0])
            history['gamma'].append(state['pitch'])
            history['roll'].append(state['roll'])
            history['slider_pos'].append(state['slider_pos'])

        # 后续绘图代码保持不变..
        fig, (ax1, ax2, ax3,ax4) =  plt.subplots(4, 1, figsize=(8, 8), sharex=True)
        
        if CTRL_MODE == 'VELOCITY':
            ax1.plot(history['t'], history['x_dot'], label='Actual $v_c$', color='orange')
            # ==== Read CSV Files
            csv_path_1 = os.path.join('csv','simulation_VELOCITY_Model_motorDamp_Controller.csv') # initial angle = 60˚
            csv_data_1 = pd.read_csv(csv_path_1)
            csv_path_2 = os.path.join('csv','simulation_VELOCITY_Model_nonLinear_Controller.csv')
            csv_data_2 = pd.read_csv(csv_path_2)  
            # ====          
            if csv_data_1 is not None:
                ax1.plot(csv_data_1['time'], csv_data_1['x_c_dot'], ls='--', label='motorDamp Reference', color='green')
                ax1.plot(csv_data_2['time'], csv_data_2['x_c_dot'], ls='--', label='nonLinear Reference', color='purple')

            ax1.axhline(y=TARGET_VX, color='red', ls='--', label=f'Target Vel={TARGET_VX}')
            ax1.set_ylabel("Velocity (m/s)")
            ax1.set_ylim([-0.5,TARGET_VX+1])

            ax2.plot(history['t'], np.degrees(history['gamma']), color='orange')
            ax2.plot(csv_data_1['time'], csv_data_1['gamma']*180/np.pi, ls='--', label='motorDamp Reference', color='green')
            ax2.plot(csv_data_2['time'], csv_data_2['gamma']*180/np.pi, ls='--', label='nonLinear Reference', color='purple')

        else:
            ax1.plot(history['t'], history['x'], label='Actual $x_c$', color='orange')
            if CTRL_MODE == 'POSITION':
                ax1.axhline(y=TARGET_X, color='red', ls='--', label=f'Target={TARGET_X}')
                # ==== Read CSV Files
                csv_path_1 = os.path.join('csv','simulation_POSITION_Model_motorDamp_Controller.csv')
                csv_data_1 = pd.read_csv(csv_path_1)
                csv_path_2 = os.path.join('csv','simulation_POSITION_Model_nonLinear_Controller.csv')
                csv_data_2 = pd.read_csv(csv_path_2)  
                # ====
                if csv_data_1 is not None:
                    # ax1.plot(csv_data_1['time'], csv_data_1['x_c'], ls='--', label='motorDamp Reference', color='green')
                    # ax1.plot(csv_data_2['time'], csv_data_2['x_c'], ls='--', label='nonLinear Reference', color='purple')
                    
                    ax2.plot(history['t'], np.degrees(history['gamma']), color='orange')
                    # ax2.plot(csv_data_1['time'], csv_data_1['gamma']*180/np.pi, ls='--', label='motorDamp Reference', color='green')
                    # ax2.plot(csv_data_2['time'], csv_data_2['gamma']*180/np.pi, ls='--', label='nonLinear Reference', color='purple')
                    # ax1.set_ylabel("Position (m)")
                #ax1.set_ylim([-0.5,TARGET_X+1])
                    ax3.plot(history['t'], np.degrees(history['roll']), label='Roll')
                    ax3.axhline(0, linestyle='--', linewidth=1)
                    ax3.set_xlabel('Time [s]')
                    ax3.set_ylabel('Roll [deg]')
                    ax3.set_title('Roll Angle')
                    ax3.legend()
                    ax3.grid(True)

                    ax4.plot(history['t'],history['slider_pos'],label='Slider Position')
                    ax4.set_xlabel('Time [s]')
                    ax4.set_ylabel('Slider Position [m]')
                    ax4.legend()
                    ax4.grid(True)
            elif CTRL_MODE == 'BALANCE':
                ax2.plot(history['t'], np.degrees(history['gamma']), color='orange')
                ax3.plot(history['t'], np.degrees(history['roll']), label='Roll')
                ax3.set_xlabel('Time [s]')
                ax3.set_ylabel('Roll [deg]')
                ax3.set_title('Roll Angle')
                ax3.legend()
                ax3.grid(True)
                ax4.plot(history['t'],history['slider_pos'],label='Slider Position')
                ax4.set_xlabel('Time [s]')
                ax4.set_ylabel('Slider Position [m]')
                ax4.legend()
                ax4.grid(True)


        ax1.set_title(f"Convergence Analysis ({CTRL_MODE} MODE)"); ax1.grid(True); ax1.legend()
        ax1.set_xlim([0, SIM_DURATION])
        
        
        ax2.set_ylabel("Absolute Angle (deg)"); ax2.set_xlabel("Time (s)"); ax2.grid(True)
        ax2.set_xlim([0, SIM_DURATION])

        plt.tight_layout()
        plt.savefig('plot_result.png')
        print("Done. The results is saved as plot_result.png。")
        try:
            subprocess.run(['open', 'plot_result.png']) 
        except:
            pass 

if __name__ == "__main__":
    main()