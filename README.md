# MIMO-Monotone Neural Network for Voltage Control

Welcome to the unofficial repository for the MIMO (Multiple Input, Multiple Output) Monotone Neural Network project, aimed at optimizing voltage control. This is an ongoing project.
## Project Structure

The repository is structured as follows:

- `config.txt`: Contains configuration settings for main packages, ensuring a seamless setup.
- `icnn.py`: Implements the Input Convex Neural Network (ICNN), a cornerstone of our voltage control approach.
- `distributed_controller.py`: Defines the MIMO Monotone Neural Network, central to distributed voltage control strategies.
- `safe_ddpg.py`: Outlines the Deep Deterministic Policy Gradient (DDPG) algorithm used for controller optimization.
- `env_13bus.py`: Sets up the simulation environment, based on a 13-bus system, for training and testing.
- `train.py`: Facilitates model training for voltage control using reinforcement learning methods.
- `test.py`: Assesses the model's performance and verifies its monotonicity properties.

## Getting Started

To begin training the model for voltage control:

```bash
python train.py
```

For performance testing and verification of monotonicity:

```bash
python test.py
```

This project integrates the Pandapower model for realistic simulation environments and includes checkpoint support for progress saving and recovery.