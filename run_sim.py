from feedback_loop_simulator import FeedbackLoopFairnessDegradationSimulator
from sklearn.ensemble import RandomForestClassifier
from fairlearn.datasets import fetch_adult
from feedback_loop_simulator import FeedbackLoopFairnessDegradationSimulator

data = fetch_adult()
df = data.frame
model = RandomForestClassifier(random_state=42)

sim = FeedbackLoopFairnessDegradationSimulator(
    model=model,
    dataset=df,
    target_variable="class",
    sensitive_variables=['sex', 'race']
)

results = sim.run_simulation(n_iterations=3)