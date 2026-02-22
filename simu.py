import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import subprocess

# ==========================================
# 1. Config
# ==========================================
# Operation mode: 
# 'VIEWER' - Mujoco UI
# 'PLOT'   - Generate plot 
RUN_MODE = 'PLOT'       # 'PLOT', 'VIEWER'
CTRL_MODE = 'VELOCITY'  # 'POSITION', 'BALANCE', 'VELOCITY'
SIM_DURATION = 30.0     
TARGET_POS = 2.0        
TARGET_VEL = 1.0        

# ==========================================
# 2. Parameters
# ==========================================
class Params:

    K_gamma = 20.0      
    K_dgamma = 2.0       

    K_position = 0.5   
    K_velocity = 2    
    
    max_tilt = (80/180)*np.pi

# ==========================================
# 3. State Extraction (Abs Coordinate)
# ==========================================
def get_abs_state(data):
    # Abs Angle
    body_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, 'box_link')
    mat = data.xmat[body_id].reshape(3, 3)
    z_axis = mat[:, 2] 
    abs_gamma = np.arctan2(z_axis[0], z_axis[2]) 
    
    # 2-D state Extraction (only contains pitch)
    x_c = data.qpos[0]
    x_c_dot = data.qvel[0]
    
    # Abs omega
    wheel_pitch_dot = data.joint('root_y').qvel[0]
    hinge_pitch_dot = data.joint('box_hinge').qvel[0]
    abs_gamma_dot = wheel_pitch_dot + hinge_pitch_dot
    
    return [x_c, x_c_dot, abs_gamma, abs_gamma_dot]

# ==========================================
# 4. Controller
# ==========================================
class SegwayController:
    def __init__(self):
        self.last_t = 0.0

    def compute(self, state, t):
        x, x_dot, gamma, gamma_dot = state
        dt = t - self.last_t if t > self.last_t else 0.0005
        self.last_t = t

        if CTRL_MODE == 'POSITION':
            pos_err = TARGET_POS - x
            target_gamma = Params.K_position * pos_err + Params.K_velocity * (0 - x_dot)
            target_gamma = np.clip(target_gamma, -Params.max_tilt, Params.max_tilt)

        elif CTRL_MODE == 'VELOCITY':
            vel_err = TARGET_VEL - x_dot
            target_gamma = Params.K_velocity * vel_err
            target_gamma = np.clip(target_gamma, -Params.max_tilt, Params.max_tilt)

        else: # BALANCE
            target_gamma = Params.K_velocity * (0 - x_dot)
            target_gamma = np.clip(target_gamma, -Params.max_tilt, Params.max_tilt)

        gamma_err = gamma - target_gamma
        tau = Params.K_gamma * gamma_err + Params.K_dgamma * gamma_dot

        return tau


# ==========================================
# 6. Simulation
# ==========================================
def main():
    model = mujoco.MjModel.from_xml_path('Model/out.xml') 
    
    if RUN_MODE == 'TUNE':
        auto_tune(model)
        return

    data = mujoco.MjData(model)
    controller = SegwayController()
    data.joint('box_hinge').qpos[0] = 0.05 # Perturbation

    history = {'t': [], 'x': [], 'x_dot': [], 'gamma': []}

    if RUN_MODE == 'VIEWER':
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time < SIM_DURATION:
                step_start = time.time()
                
                state = get_abs_state(data)
                if abs(state[2]) > np.pi/2: 
                    print(f"The System fails at {data.time:.2f}s")
                    break 

                torque = controller.compute(state, data.time)
                data.actuator('main_motor').ctrl[0] = np.clip(torque, -150, 150)
                
                mujoco.mj_step(model, data)
                viewer.sync()
                
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    elif RUN_MODE == 'PLOT':
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        print(f"Simulating for {CTRL_MODE} ...")
        while data.time < SIM_DURATION:
            state = get_abs_state(data)
            if abs(state[2]) > np.pi/2: break 
            
            torque = controller.compute(state, data.time)
            data.actuator('main_motor').ctrl[0] = np.clip(torque, -150, 150)
            mujoco.mj_step(model, data)
            
            history['t'].append(data.time)
            history['x'].append(state[0])
            history['x_dot'].append(state[1])
            history['gamma'].append(state[2])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        if CTRL_MODE == 'VELOCITY':
            ax1.plot(history['t'], history['x_dot'], label='Actual $v_c$')
            ax1.axhline(y=TARGET_VEL, color='red', ls='--', label=f'Target Vel={TARGET_VEL}')
            ax1.set_ylabel("Velocity (m/s)")
        else:
            ax1.plot(history['t'], history['x'], label='Actual $x_c$')
            if CTRL_MODE == 'POSITION':
                ax1.axhline(y=TARGET_POS, color='red', ls='--', label=f'Target={TARGET_POS}')
            ax1.set_ylabel("Position (m)")
            
        ax1.set_title(f"Convergence Analysis ({CTRL_MODE} MODE)")
        ax1.grid(True); ax1.legend()
        
        ax2.plot(history['t'], np.degrees(history['gamma']), color='orange')
        ax2.set_ylabel("Absolute Angle (deg)"); ax2.set_xlabel("Time (s)"); ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('plot_result.png')
        print("Done. The results is saved as plot_result.png。")
        try:
            subprocess.run(['open', 'plot_result.png']) 
        except:
            pass 

if __name__ == "__main__":
    main()