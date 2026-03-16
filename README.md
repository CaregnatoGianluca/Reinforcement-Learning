# Snake Game Reinforcement Learning

This repository contains a comparative study of Reinforcement Learning algorithms applied to the classic game of Snake. The environment is evaluated under two conditions:
1. **Fully Observable**: The agent sees the entire $7 \times 7$ board.
2. **Partially Observable**: The agent only sees a $5 \times 5$ window centered on its head.

I implemented three RL agents and compared them against a Greedy Breadth-First Search (BFS) heuristic baseline:
- **Deep Q-Network (DQN)**
- **REINFORCE with Baseline**
- **Actor-Critic**

All RL agents are built on a shared Convolutional Neural Network (CNN) backbone and implemented via TensorFlow 2 / Keras.

## Repository Structure

- `snake/`
  - `agents/`: Contains the implementation of the CNN backbone (`networks.py`) and the three RL agents (`dqn_agent.py`, `reinforce_agent.py`, `actor_critic_agent.py`).
  - `environments_fully_observable.py` / `environments_partially_observable.py`: The Snake game environments.
  - `baseline.py`: Hand-crafted Greedy BFS heuristic baseline.
  - `train.py`: Script to train the models from scratch.
  - `evaluate.py`: Script to evaluate the best model (DQN) and the baseline.
  - `evaluate_all.ipynb`: A complete Jupyter Notebook that downloads all the pre-trained weights and evaluates all the combined models (Baseline, DQN, REINFORCE, Actor-Critic).
  - `utils.py`: Various helper functions.
  - `weights/`: Local folder containing the best model's default weights (DQN).
- `report/`: The \LaTeX\ source code and compiled PDF containing the full details of the project methods and results (`main2.pdf`).

## How to run it

### Evaluating the Best Model
You can directly run the evaluation script to see the performance of the **DQN agent** and the **Baseline** on 100 parallel boards for 1,000 steps.
```bash
cd snake
python evaluate.py
```
*Note: The script uses the pre-trained weights located in the `snake/weights/` folder.*

### Comprehensive Evaluation (All Models)
To evaluate all the implemented algorithms and view their smoothed learning curves, run the provided Jupyter Notebook:
```bash
cd snake
jupyter notebook evaluate_all.ipynb
```
Inside the notebook:
1. A cell will automatically download the required pre-trained weights (DQN, REINFORCE, Actor-Critic) from Google Drive into a new `snake/allWeigths/` folder (total size $\approx 8.71$ MB).
2. It evaluates all 3 agents along with the baseline.
3. It prints the evaluation metrics and plots the side-by-side learning curves inline.

### Training from Scratch
To train the agents from scratch, utilize the `train.py` script. You can pass arguments to select the agent, the environment type, and the number of training iterations.

Use `python train.py --help` for the full list of options.

## Results
- **DQN** achieved the highest average reward and the best trade-off between fruit collection and self-avoidance (exceeding the baseline by $+22\%$).
- Partial observability degraded the RL agents far less ($\approx 10\%$) compared to the BFS heuristic ($-22\%$), showing the robustness of learned policies over pure search algorithms when facing limited information.
- A full analysis of the findings can be read in `report/main2.pdf`.
