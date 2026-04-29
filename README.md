# Pendulum Control: Data Science Project with MPC and Neural Network Imitation

![Example Controller Comparison](notebooks/controller_comparison.png)

## Project Overview
This project explores control strategies for the classic `Pendulum-v1` environment using a data science workflow. The main goal is to swing up and stabilize a pendulum using:
- **Random controller** (baseline)
- **Model Predictive Control (MPC)** with evolutionary optimization
- **Neural Network (NN) controller** trained to imitate MPC

The notebook demonstrates simulation, data generation, supervised learning, imitation learning, and model evaluation.

## Problem Statement
Design and compare controllers for the pendulum environment to:
- Achieve upright stabilization
- Minimize control effort
- Demonstrate the value of data-driven imitation learning

## Approach
1. **Simulation & Data Generation**
   - Simulate episodes with random and MPC controllers
   - Collect state-action-reward trajectories
2. **Expert Demonstrations**
   - Use MPC to generate high-quality control data
   - Filter for successful episodes
3. **Exploratory Data Analysis**
   - Visualize state and action distributions
   - Analyze relationships (e.g., angle vs. torque)
4. **Neural Network Training**
   - Train a feedforward NN to predict MPC actions
   - Use train/validation/test splits and standardized features
   - Evaluate with regression metrics (MAE, RMSE, $R^2$)
5. **Controller Evaluation**
   - Compare random and NN controllers in simulation
   - Visualize score distributions and example trajectories

## Key Results & Visualizations
- **Random controller**: Unstable, poor scores
- **MPC**: High-quality but slow expert policy
- **Neural Network**: Fast, effective imitation of MPC

### Example Plots

#### 1. Controller Score Distribution
![Controller Score Distribution](notebooks/controller_score_distribution.png)

#### 2. Neural Network Training Loss
![NN Training Loss](notebooks/nn_training_loss.png)

#### 3. Predicted vs Actual Torque
![Predicted vs Actual Torque](notebooks/pred_vs_actual_torque.png)

#### 4. Example Trajectory
![Example Trajectory](notebooks/example_trajectory.png)

*Replace the above image paths with your actual plot outputs.*

## How to Run
1. Install dependencies:
   ```bash
   pip install gymnasium scipy skorch torch scikit-learn pandas matplotlib
   ```
2. Open the notebook in `notebooks/pendulum_mpc_nn_portfolio_scipy.ipynb` and run all cells.
3. Plots are generated inline; save them as images for your report or README.

## Repository Structure (Suggested)
```
pendulum-control-data-science/
├── notebooks/
│   └── pendulum_mpc_nn_portfolio_scipy.ipynb
├── src/
│   ├── controllers.py
│   ├── simulation.py
│   └── models.py
├── data/
│   └── .gitkeep
├── requirements.txt
├── README.md
└── .gitignore
```

## References
- [OpenAI Gymnasium: Pendulum-v1](https://gymnasium.farama.org/environments/classic_control/pendulum/)
- [PyTorch](https://pytorch.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Skorch](https://skorch.readthedocs.io/)

---
*This project demonstrates how optimization-based controllers can generate expert data for imitation learning, and how a neural network can learn a fast, effective policy. For further improvements, see the notebook's "Future work" section.*
