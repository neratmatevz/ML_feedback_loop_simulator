# ML Feedback Loop Fairness Degradation Simulator

A Python tool for simulating and visualizing how machine learning feedback loops affect model performance and fairness over time.

## What does it do?

In real-world ML systems, model predictions often influence future training data. For example:
- A hiring algorithm's decisions affect who gets hired, which becomes the "ground truth" for future models
- A loan approval system's predictions determine who receives loans, shaping future default data

This creates **feedback loops** where biased predictions reinforce themselves over time, potentially amplifying unfairness.

This simulator demonstrates this by:
1. Training a model on initial data
2. Using predictions as "true" labels for the next iteration
3. Repeating across multiple iterations
4. Tracking how performance and fairness metrics degrade

## Features

- **Supports Classification & Regression**: Automatically detects task type
- **Runs Simulation**: Runs simulation based on inputed iterations 
- **Fairness Metrics**: Tracks group-level performance disparities
  - Classification: Accuracy, F1, Precision, Recall, Selection Rate, TPR, FNR
  - Binary Classification: DPD (Demographic Parity Difference), EOD (Equalized Odds Difference)
  - Regression: MAE, MSE, R², Mean/Std Residuals
- **Visualization**: Per-iteration plots and summary trend charts
- **More Ways To Execute**: Supports Jupyter Notebook (e.g.:MLFeedbackLoopFairDegSimulator.ipynb) and normal execution (e.g.:run_sim.py)

## Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn fairlearn
```

## Quick Start

```python
from sklearn.ensemble import RandomForestClassifier
from feedback_loop_simulator import FeedbackLoopFairnessDegradationSimulator
import pandas as pd
from fairlearn.datasets import fetch_adult

# Load your dataset
df = pd.read_csv("your_data.csv")
# OR use an existing one to test
data = fetch_adult()
df = data.frame

# Create simulator
sim = FeedbackLoopFairnessDegradationSimulator(
    model=RandomForestClassifier(random_state=42),
    dataset=df,
    target_variable="target_column",
    sensitive_variables=["gender", "race"]
)

# Run simulation with 5 iterations
sim.run_simulation(n_iterations=5)
```

## Usage Options

### Hide iteration details (show only summary)

```python
sim.set_show_iteration_data(False)
sim.run_simulation(n_iterations=10)
```

### Use custom display function (e.g., in Jupyter)

```python
from IPython.display import display

sim = FeedbackLoopFairnessDegradationSimulator(
    model=model,
    dataset=df,
    target_variable="target",
    sensitive_variables=["gender"],
    display_func=display  # Pretty tables in Jupyter
)
```

### Use slider to run the simulation (e.g., in Jupyter)

```python
from ipywidgets import IntSlider, interact

slider = IntSlider(
    value=3,                    # Set starting value of iterations
    min=3,                      # Set minimum number of iterations
    max=100,                    # Set maximum number of iterations
    step=1,                     # Value incremented by amount of steps
    description="Iterations",   
    continuous_update=False     # Restrict execution to mouse up event
)

_ = interact(sim.run_simulation, n_iterations=slider)
```
## Example Notebook

See `MLFeedbackLoopFairDegSimulator.ipynb` for complete examples including:
- Binary classification (Adult dataset)
- Multiclass classification (synthetic data with built-in bias)
- Regression (Boston housing dataset)

## How It Works

```
Iteration 1: Data[0] → Train → Data[1] → Predict → Save predictions
Iteration 2: Data[1] + predictions → Train → Data[2] → Predict → Save predictions
Iteration 3: Data[2] + predictions → Train → Data[3] → Predict → ...
```

Each iteration uses the previous iteration's predictions as training labels, simulating how feedback loops corrupt ground truth over time.
