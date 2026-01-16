"""
Example: Running the ML Feedback Loop Simulator without Jupyter Notebook

This script demonstrates how to use the FeedbackLoopFairnessDegradationSimulator
in a standalone Python environment (terminal, IDE, or script).

For interactive notebook examples, see: MLFeedbackLoopFairDegSimulator.ipynb

Usage:
    python run_sim.py
"""

from feedback_loop_simulator import FeedbackLoopFairnessDegradationSimulator
from sklearn.ensemble import RandomForestClassifier
from fairlearn.datasets import fetch_adult

data = fetch_adult()
df = data.frame
model = RandomForestClassifier(random_state=42)

# Initialize the simulator, without display function; print is the default display function
sim = FeedbackLoopFairnessDegradationSimulator(
    model=model,
    dataset=df,
    target_variable="class",
    sensitive_variables=['sex', 'race']
)

sim.set_show_iteration_data(False)

results = sim.run_simulation(n_iterations=3)