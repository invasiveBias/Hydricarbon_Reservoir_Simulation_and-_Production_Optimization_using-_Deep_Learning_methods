# Deep Learning for Hydrocarbon Reservoir Simulation and Production Optimization

As the research focus for my undergraduate honors thesis in Geology, this project explores the application of deep learning techniques for hydrocarbon reservoir management. It is divided into two main parts: first, simulating reservoir behavior using a finetuned language model, and second, optimizing production rates using Deep Reinforcement Learning (DRL) guided by the trained simulator.

Managing hydrocarbon reservoirs effectively is crucial for maximizing recovery and economic value. Traditional simulation methods can be computationally expensive. This project investigates using data-driven deep learning models as surrogate simulators and reinforcement learning for intelligent production control.

## Project Structure

The project is structured into two interdependent components:

1.  **Hydrocarbon Reservoir Simulation:** Developing a data-driven model to predict well-level production and injection rates based on geological, geometric, and operational parameters.
2.  **Production Optimization:** Training a Deep Reinforcement Learning agent to determine optimal operational parameters (specifically, injection pressure) to maximize economic value (Gross Present Value).

## Part 1: Hydrocarbon Reservoir Simulation

The goal of this part is to create a fast, data-driven surrogate model for hydrocarbon reservoir behavior at the well level.

### Approach

A finetuning approach was employed using the **Distil-BERT** model, a smaller, faster variant of the BERT family. Distil-BERT was chosen for its state-of-the-art efficiency in feature representation combined with a size suitable for training and inference on limited resources (specifically, a Kaggle T4 GPU).

The finetuning process involved:

* Loading a pretrained Distil-BERT model.
* Freezing the weights of the pretrained layers to retain their learned representations.
* Replacing the original output layer with a custom **Multi-head Regressor**.
* This regressor has four output heads designed to predict the following continuous values for each well at a given timestep:
    * Gas Production Rate ($m^3/day$)
    * Oil Production Rate ($m^3/day$)
    * Water Production Rate ($m^3/day$)
    * Water Injection Rate ($m^3/day$)
* Training the multi-head regressor (with randomly initialized weights) to map the learned representations from the frozen Distil-BERT layers to the target production and injection rates.

### Data and Features

The model was trained on a preprocessed variant of the **Oil Reservoir Simulation Dataset (ORSD)** obtained from the IBM Data Asset Exchange platform. The dataset provides well-level data over time.

The input features used to train the simulator model are:

* **Geological:**
    * Porosity ($\phi$)
    * Horizontal Permeability ($k_h$)
    * Vertical Permeability ($k_v$)
* **Geometric:**
    * X-coordinate ($x$)
    * Y-coordinate ($y$)
    * Z-coordinate ($z$)
* **Operational/Production:**
    * Well Type (e.g., producer, injector)
    * Injected Pressure ($P_{inject}$)

The model was trained using a **Mean Squared Error (MSE)** loss function between the predicted and actual output rates.

## Part 2: Production Optimization using Deep Reinforcement Learning

This part focuses on training an intelligent agent to make decisions on optimal operational parameters to maximize economic returns from the reservoir.

### Approach

A **Deep Reinforcement Learning (DRL)** approach was used, specifically the **Proximal Policy Optimization (PPO)** algorithm. PPO is a policy gradient method known for its balance of sample efficiency and stable policy updates, making it suitable for problems with continuous action spaces like predicting injection pressure.

A custom DRL environment was built to facilitate the training of the PPO agent.

* **Agent State Observation:** At each timestep, the agent observes a state vector comprising 7 features:
    * Geological: Porosity, Horizontal Permeability, Vertical Permeability
    * Geometric: X, Y, Z coordinates
    * Operational: Well Type
* **Agent Action Space:** The agent's action is a single continuous value: the **Injected Pressure ($P_{inject}$)** for the wells it controls.
* **Environment Dynamics (Simulator Integration):** The agent's chosen action ($P_{inject}$) is combined with the observed 7 features, forming the 8 input features required by the **trained Reservoir Simulation model** from Part 1. The simulator then predicts the four production/injection rates (Gas, Oil, Water Production; Water Injection).
* **Reward Function:** The economic value of the predicted rates is calculated to serve as the reward for the agent. This calculation considers:
    * Predicted Gas, Oil, and Water Production Rates
    * Water Injection Rate
    * Oil Price
    * Gas Price
    * Production Costs
    * Water Injection Costs
    * This information is used to estimate the **Gross Present Value (GPV)** at the current timestep. The GPV estimation forms the reward signal, encouraging the agent to learn policies that maximize economic value.
* **Training:** The PPO agent is trained to maximize this GPV-based reward signal, using the PPO loss function to update its policy network.

### Scalability
To model a larger production field rather than just a single well, the training environment was **vectorized**. This allows the DRL agent to train simultaneously on data from **25 different wells**, significantly improving training efficiency and the agent's ability to generalize across different well characteristics.

## Dataset

The project utilizes a personally preprocessed version of the **Oil Reservoir Simulation Dataset (ORSD)**.

* **Source:** IBM Data Asset Exchange platform.
* **Original Dataset Overview:** [https://dax-cdn.cdn.appdomain.cloud/dax-oil-reservoir-simulations/1.0.0/data-preview/Part%201%20-%20Dataset%20overview.html](https://dax-cdn.cdn.appdomain.cloud/dax-oil-reservoir-simulations/1.0.0/data-preview/Part%201%20-%20Dataset%20overview.html)
