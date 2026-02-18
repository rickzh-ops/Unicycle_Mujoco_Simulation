import mujoco
import mujoco.viewer
import numpy as np
import time


model = mujoco.MjModel.from_xml_path("Model/out.xml")
data  = mujoco.MjData(model)


body_name = "wheel" 
bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

if bid == -1:
    raise RuntimeError(f"can't find '{body_name}' Body, check XML!")

F = np.array([10, 0, 0.0])           #give the wheel a push
r = np.array([0.0, 0.0, 0.25])     
tau = np.cross(r, F)               

t_start = 2
t_end   = 2.2

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        
        data.xfrc_applied[bid, :] = 0.0

        if t_start <= data.time <= t_end:
            data.xfrc_applied[bid, 0:3] = F
            data.xfrc_applied[bid, 3:6] = tau
            
            # print(f"Applying force at time: {data.time:.3f}")

        mujoco.mj_step(model, data)

        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)