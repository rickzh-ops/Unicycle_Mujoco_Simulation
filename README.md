# Unicycle Mujoco  Simulation

## 📑 Table of Contents
1. [Environment Setup](#i-mujoco-environment-setup-guide)
2. [Simulation Documentation](#ii-simulation-code)
<br>
<br>


# I. MuJoCo Environment Setup Guide

This section provides step-by-step instructions to set up a physics simulation environment using MuJoCo and Conda.

## Tbale of Contents
1. [Prerequisites](#1-prerequisites)

2. [Install Conda (Miniconda)](#2-install-conda-miniconda)

3. [Create Virtual Environment](#3-create-virtual-environment)

4. [Install MuJoCo](#4-install-mujoco)

5. [Verification](#5-verification)

6. [Troubleshooting](#6-troubleshooting)

## 1. Prerequisites
Ensure your system meets the following requirements:
- OS: Windows 10/11, Linux (Ubuntu 20.04+ recommended), or macOS.
- GPU: Recommended for hardware acceleration (ensure drivers are up to date).

## 2. Install Conda (Miniconda)
If you do not have Conda installed, we recommend Miniconda for a lightweight experience.

### For Windows
1. Download the [Miniconda Windows Installer](https://www.anaconda.com/docs/getting-started/miniconda/install).

2. Run the `.exe` file.

3. **Important**: During installation, check the box **"Add Miniconda3 to my PATH environment variable"** if you want to use it in the standard Command Prompt. Otherwise, use the "Anaconda Prompt" from the Start Menu.

### For Linux (Ubuntu)
Open your terminal and run:

```Bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Follow the prompts, then restart your terminal or run ``source ~/.bashrc``.

## 3. Create Virtual Environment
Avoid conflicts with other Python projects by creating a dedicated environment.
```Bash
# Create a new environment named 'mujoco_env'
conda create -n .venv python=3.11 -y

# Activate the environment
conda activate .venv
```
## 4. Install MuJoCo
Since MuJoCo became open-source (version 2.1.1+), it can be installed easily via ``pip``.

```Bash
# Install the core MuJoCo library
pip install mujoco

# (Optional) Install Gymnasium for RL tasks
pip install gymnasium[mujoco]
```

## 5. Verification
Create a file named ``check_install.py`` and paste the following code:
```Python
import mujoco
import numpy as np

# Create a simple model
xml = """
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="sphere" size=".1" rgba="0 .9 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_step(model, data)

print(f"Success! Sphere position: {data.qpos[:3]}")
```
Run the script:
```Bash
python check_install.py
```
If it prints a position without errors, your installation is active!

## 6. Troubleshooting
- Linux GL Errors: If you see GLEW initialization error, install OpenGL dependencies:
```Bash
sudo apt update
sudo apt install libgl1-mesa-dev libglew-dev
```
- Environment Not Found: Ensure you have activated the environment using ``conda activate .venv`` before running your scripts.

# II. Simulation Code

This section introduces the files used for simulation in this repository.

## 1. The Model
The key simulation model is defined in ``Model\out.xml`` .

To inspect the model and its physical properties, execute the following command:
```Bash
mjpython view.py
```
Once the MuJoCo viewer initializes, the simulation will run as follows:

- Initial State: The model remains static for the first 2 seconds.

- External Force: After this 2-second delay, a force is automatically applied to the model.

- Movement: This force initiates the rotation, and the wheel will begin rolling.

## 2. Simulation Code
### 2.1 Configuration
You can modify the following parameters in the "Config" section of ``simu.py``:

- ``RUN_MODE``:

  - ``'VIEWER'``: Launches the interactive MuJoCo passive viewer for real-time 3D observation.

  - ``'PLOT'``: Runs the simulation in the background and generates a ``plot_result.png`` file for data analysis.

- ``CTRL_MODE``:

  - ``'POSITION'``: The vehicle moves toward and stays at TARGET_POS.

  - ``'VELOCITY'``: The vehicle maintains a constant speed defined by TARGET_VEL.

  - ``'BALANCE'``: The vehicle maintains its current position and suppresses any velocity drift.

### 2.2 Controller Architecture

The system utilizes a dual-loop control strategy:
1. **Outer Loop (Kinematics)**: Processes position or velocity errors to calculate a "Target Tilt Angle" ($target\_gamma$). This angle is clipped by ``max_tilt`` to prevent the inner loop from reaching an unrecoverable state.
2. **Inner Loop (Dynamics)**: High-frequency feedback loop that tracks the $target\_gamma$ by calculating the motor torque ($\tau$) based on the current absolute angle and angular velocity.

### 2.3 Running the Simulation
Ensure your virtual environment is activated:
```Bash
conda activate .venv
```

Execute the simulation using ``mjpython`` to ensure proper 3D rendering and library support:
```Bash
mjpython simu.py
```

### 2.4 Data Visualization
When ``RUN_MODE`` is set to ``'PLOT'``, the script outputs two synchronized graphs:
- **Upper Plot**: Tracks the convergence of $x_c$ (Position) or $v_c$ (Velocity) against the red dashed target line.
- **Lower Plot**: Displays the absolute tilt angle in degrees to monitor the "lean" of the unicycle during acceleration and braking.