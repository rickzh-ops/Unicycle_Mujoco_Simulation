# MuJoCo Environment Setup Guide

This guide provides step-by-step instructions to set up a physics simulation environment using MuJoCo and Conda.

## Tbale of Contents
1. Prerequisites

2. Install Conda (Miniconda)

3. Create Virtual Environment

4. Install MuJoCo

5. Verification

6. Troubleshooting

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