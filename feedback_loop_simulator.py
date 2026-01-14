import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score,
    recall_score,
    mean_squared_error, 
    mean_absolute_error,
    r2_score
)
from sklearn.utils.multiclass import type_of_target

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    equalized_odds_difference,
    true_positive_rate,
    false_negative_rate,
)

warnings.filterwarnings('ignore', category=FutureWarning)

class FeedbackLoopFairnessDegradationSimulator:
    """
    Simulates how model performance and fairness change over time because of ML feedback loops.

    Supports:
    - Classification: performance and fairness metrics by group
    - Regression: error and residual-based disparity metrics
    """

    def __init__(self, model, dataset, target_variable, sensitive_variables, display_func=None):
        self.base_model = model
        self.data = dataset.reset_index(drop=True).copy()
        self.target = target_variable
        self.sensitive_vars = sensitive_variables
        self.sensitive_data_original = self.data[self.sensitive_vars].copy()
        self.display_func = display_func if display_func else print
        self.show_iteration_plots = True
        self._preprocess_data()
        self.is_classification = self._detect_task_type()
        self.all_results = []


    # ------------------------------------------------------------------
    # Setter for showing iteration plots
    # ------------------------------------------------------------------
    def set_show_iteration_plots(self, show):
        """
        Set whether to display plots for each iteration.
        Useful with higher number of iterations to reduce output length,
        """
        self.show_iteration_plots = show


    # ------------------------------------------------------------------
    # Preprocessing data
    # ------------------------------------------------------------------
    def _preprocess_data(self):
        """
        Prepare the dataset for modeling and fairness analysis.

        Steps:
        1. Encode the target variable if it is categorical
        2. One-hot encode non-sensitive categorical features
        3. Encode sensitive variables for modeling while keeping originals for fairness evaluation
        """

        # Store for printing preprocessing steps
        self.target_encoder = None
        
        print(f"{'='*150}")
        print(f"Data Preprocessing")
        print(f"{'='*150}")
        print()

        # Encode target var if categorical
        if self.data[self.target].dtype == 'object' or isinstance(self.data[self.target].iloc[0], str):
            self.target_encoder = LabelEncoder()
            self.data[self.target] = self.target_encoder.fit_transform(self.data[self.target])
            print(f"Target variable '{self.target}' encoded: {dict(zip(self.target_encoder.classes_, self.target_encoder.transform(self.target_encoder.classes_)))}")
        
        # Identify categorical variables without target and sensitive variables
        categorical_cols = []
        for col in self.data.columns:
            if col != self.target and col not in self.sensitive_vars:
                if self.data[col].dtype == 'object' or self.data[col].dtype.name == 'category':
                    categorical_cols.append(col)
        
        # One-hot encode categorical features
        if categorical_cols:
            print(f"One-hot encoding categorical features: {categorical_cols}")
            self.data = pd.get_dummies(
                self.data, 
                columns=categorical_cols, 
                drop_first=True,
                dtype=int
            )
        
        # Encode sensitive variables for use as features (keep originals for fairness metrics)
        for s_var in self.sensitive_vars:
            if self.data[s_var].dtype == 'object' or isinstance(self.data[s_var].iloc[0], str):
                encoder = LabelEncoder()
                self.data[f"{s_var}_encoded"] = encoder.fit_transform(self.data[s_var])
                print(f"Sensitive variable '{s_var}' encoded for features: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")
        
        print(f"Dataset shape after preprocessing: {self.data.shape}")


    # ------------------------------------------------------------------
    # Detect model task type: Classification vs Regression
    # ------------------------------------------------------------------
    def _detect_task_type(self):
        """
        Detect whether the task is classification or regression 
        based on the target variable. 
        """
        y = self.data[self.target]
        task_type = type_of_target(y)

        print(f"\n{'='*150}")
        print(f"Detect Model Task Type")
        print(f"{'='*150}")
        print()

        print(f"Detected task type: {task_type} classification" if task_type in ["binary", "multiclass"] else f"Detected task type: {task_type} regression")

        return task_type in ["binary", "multiclass"]


    # ------------------------------------------------------------------
    # Data splitting based on iterations
    # ------------------------------------------------------------------
    def data_split(self, n_iterations):
        """
        Split the dataset into sequential chunks for simulation.

        The data is divided into (n_iterations + 1) parts so that
        each iteration trains on one part and tests on the next.
        """
        n_parts = n_iterations + 1
        return np.array_split(self.data, n_parts)


    # ------------------------------------------------------------------
    # Calculate performance and fairness metrics for iteration
    # ------------------------------------------------------------------
    def _compute_metrics_by_group(self, y_true, y_pred, sensitive_series):
        """
        Compute performance and fairness metrics separately
        for each sensitive group.

        - Classification:
            * Accuracy, F1, Precision, Recall
            * Selection Rate, TPR, FNR (binary only)
        - Regression:
            * MAE, MSE, R²
            * Mean and standard deviation of residuals
        """
        if self.is_classification:
            # Check if binary or multiclass
            n_classes = len(np.unique(y_true))
            is_binary = n_classes == 2
            
            if is_binary:
                # Metrics for binary classification
                metrics = {
                    "accuracy": accuracy_score,
                    "f1_score": lambda y_t, y_p: f1_score(y_t, y_p, average="binary"),
                    "precision": lambda y_t, y_p: precision_score(y_t, y_p, average="binary", zero_division=0),
                    "recall": lambda y_t, y_p: recall_score(y_t, y_p, average="binary", zero_division=0),
                    "selection_rate": selection_rate,
                    "TPR": true_positive_rate,
                    "FNR": false_negative_rate,
                }
            else:
                # Metrics for multiclass classification
                metrics = {
                    "accuracy": accuracy_score,
                    "f1_score": lambda y_t, y_p: f1_score(y_t, y_p, average="weighted", zero_division=0),
                    "precision": lambda y_t, y_p: precision_score(y_t, y_p, average="weighted", zero_division=0),
                    "recall": lambda y_t, y_p: recall_score(y_t, y_p, average="weighted", zero_division=0),
                }
        else:
            # Metrics for regression
            def mean_residual(y_t, y_p):
                """Average prediction error (bias indicator)."""
                return np.mean(y_t - y_p)
            
            def std_residual(y_t, y_p):
                """Spread of prediction errors (consistency indicator)."""
                return np.std(y_t - y_p)
            
            metrics = {
                "mae": mean_absolute_error,
                "mse": mean_squared_error,
                "r2": r2_score,
                "mean_residual": mean_residual,
                "std_residual": std_residual,
            }

        # Compute metrics by sensitive group
        mf = MetricFrame(
            metrics=metrics,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_series
        )

        return mf.by_group


    # ------------------------------------------------------------------
    # Compute DPD and EOD for iteration
    # ------------------------------------------------------------------
    def _compute_aggregate_fairness(self, y_true, y_pred, sensitive_series):
        """
        Compute aggregate fairness metrics for binary classification.

        - Demographic Parity Difference (DPD)
        - Equalized Odds Difference (EOD)
        """
        n_classes = len(np.unique(y_true))
        is_binary = n_classes == 2
        
        if not self.is_classification or not is_binary:
            return None, None
        
        dpd = demographic_parity_difference(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_series
        )
        
        eod = equalized_odds_difference(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_series
        )
        
        return dpd, eod


    # ------------------------------------------------------------------
    # Simulation runner
    # ------------------------------------------------------------------
    def run_simulation(self, n_iterations):
        """
        Run the feedback-loop fairness simulation.

        For each iteration:
        1. Split data sequentially
        2. Train the model on current data
        3. Replace future training labels with past predictions (simulating feedback loops)
        4. Evaluate on the next data split
        5. Compute and visualize group-level fairness metrics
        6. Store results for summary visualization
        """
        self.all_results = []
        partitions = self.data_split(n_iterations)

        prev_predictions = None

        print(f"\n{'='*150}")
        print(f"Starting Fairness Degradation Simulation with {n_iterations} Iterations")
        print(f"{'='*150}")
        print()

        for i in range(n_iterations):
            train_df = partitions[i].copy()
            test_df = partitions[i + 1].copy()

            # Replace target with predictions from previous iteration
            if prev_predictions is not None:
                # Store original target for comparison/metrics
                train_df["original_target"] = train_df[self.target].copy()
                # Replace target with previous predictions
                train_df[self.target] = prev_predictions

            # Prepare features (remove target, original target, original sensitive vars - encoded versions remain)
            X_train = (
                train_df
                .drop(columns=[self.target, "original_target"] + self.sensitive_vars, errors="ignore")
                .select_dtypes(include=[np.number])
            )

            y_train = train_df[self.target]  # Now uses predictions as target

            X_test = (
                test_df
                .drop(columns=[self.target] + self.sensitive_vars, errors="ignore")
                .select_dtypes(include=[np.number])
            )

            y_test = test_df[self.target]

            # train a new model clone
            model = clone(self.base_model)
            model.fit(X_train, y_train)

            # Predict and store predictions for next iteration
            y_pred = model.predict(X_test)
            prev_predictions = y_pred

            print(f"\n{'*'*90}")
            print(f"Iteration {i+1}/{n_iterations}")
            print(f"{'*'*90}")

            # Store results for this iteration
            iteration_result = {
                "iteration": i + 1,
                "fairness": {},
                "aggregate_fairness": {},
            }

            # Compute and visualize fairness metrics for each sensitive variable
            for s in self.sensitive_vars:
                grouped_metrics = self._compute_metrics_by_group(
                    y_test, y_pred, test_df[s]
                )

                iteration_result["fairness"][s] = grouped_metrics

                # Compute DPD and EOD (binary classification only)
                demo_parity, eqo = self._compute_aggregate_fairness(y_test, y_pred, test_df[s])
                
                if demo_parity is not None and eqo is not None:
                    # Store aggregate metrics
                    if s not in iteration_result["aggregate_fairness"]:
                        iteration_result["aggregate_fairness"][s] = {}
                    
                    iteration_result["aggregate_fairness"][s]["DPD"] = demo_parity
                    iteration_result["aggregate_fairness"][s]["EOD"] = eqo


                print(f"\nSensitive variable: {s} (Iteration {i+1}/{n_iterations})")
                
                # Display group metrics table
                self.display_func(grouped_metrics)
                print()
                
                # Display DPD and EOD table (binary classification only)
                if demo_parity is not None and eqo is not None:
                    aggregate_df = pd.DataFrame({
                        "DPD": [demo_parity],
                        "EOD": [eqo]
                    }, index=[s])
                    self.display_func(aggregate_df)

                if self.show_iteration_plots:
                    self.plot_iteration_results(grouped_metrics, s, i + 1, demo_parity, eqo)

            self.all_results.append(iteration_result)

        # After all iterations, show summary over time
        self.plot_summary_results()


    # ------------------------------------------------------------------
    # Iteration results plotting
    # ------------------------------------------------------------------
    def plot_iteration_results(self, df, sensitive_var, iteration, dpd=None, eod=None):
        """
        Visualize fairness metrics for a single iteration.

        - Creates bar charts for each metric by sensitive group
        - Optionally displays Demographic Parity Difference (DPD)
        and Equalized Odds Difference (EOD) for binary classification
        """
        df_plot = df.reset_index().melt(
            id_vars=sensitive_var,
            var_name="metric",
            value_name="value",
        )

        metrics = df_plot["metric"].unique()
        n_metrics = len(metrics)

        # Calculate figure dimensions - max 4 metrics per row
        cols_per_row = 4
        n_metric_rows = (n_metrics + cols_per_row - 1) // cols_per_row 
        
        metric_width = 4 
        total_width = metric_width * cols_per_row  
        
        if dpd is not None and eod is not None:
            # Multi-row layout: metrics in rows of 4, DPD/EOD at bottom left
            total_rows = n_metric_rows + 1 
            fig = plt.figure(figsize=(total_width, 3.5 * total_rows))
            
            # Create gridspec with total_rows and cols_per_row columns
            gs = fig.add_gridspec(total_rows, cols_per_row,
                                  hspace=0.5, wspace=0.35,
                                  left=0.06, right=0.94, top=0.92, bottom=0.08)
            
            # Create axes for metrics
            axes = []
            for idx in range(n_metrics):
                row = idx // cols_per_row
                col = idx % cols_per_row
                axes.append(fig.add_subplot(gs[row, col]))
            
            # Bottom row: DPD/EOD takes 1 column, positioned left
            ax_aggregate = fig.add_subplot(gs[n_metric_rows, 0])
        else:
            # No DPD/EOD - just metric rows
            fig = plt.figure(figsize=(total_width, 3.5 * n_metric_rows))
            gs = fig.add_gridspec(n_metric_rows, cols_per_row,
                                  hspace=0.5, wspace=0.35,
                                  left=0.06, right=0.94, top=0.92, bottom=0.08)
            
            axes = []
            for idx in range(n_metrics):
                row = idx // cols_per_row
                col = idx % cols_per_row
                axes.append(fig.add_subplot(gs[row, col]))
            
        fig.suptitle(
            f"Iteration {iteration} – Sensitive variable: {sensitive_var}",
            fontsize=14,
            fontweight='bold'
        )

        # Plot per-group metrics
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            metric_data = df_plot[df_plot["metric"] == metric]
            bars = sns.barplot(
                data=metric_data,
                x=sensitive_var,
                y="value",
                hue=sensitive_var,
                ax=ax,
                palette="Set2",
                legend=False
            )
            ax.set_title(metric, fontsize=12)
            ax.set_xlabel("")
            ax.set_ylabel("Value", fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            
            for bar_idx, bar in enumerate(ax.patches):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height / 1.1,
                    f'{height:.2f}',
                    ha='center',
                    va='center',
                    fontsize=9,
                    fontweight='bold',
                    color='black'
                )

        # DPD and EOD subplot (for binary classification)
        if dpd is not None and eod is not None:
            aggregate_data = pd.DataFrame({
                "Metric": ["DPD", "EOD"],
                "Value": [dpd, eod]
            })
            
            sns.barplot(
                data=aggregate_data,
                x="Metric",
                y="Value",
                hue="Metric",
                ax=ax_aggregate,
                palette=["#e74c3c", "#3498db"],
                legend=False
            )
            ax_aggregate.set_title("DPD & EOD", fontsize=12)
            ax_aggregate.set_xlabel("")
            ax_aggregate.set_ylabel("Value", fontsize=10)
            
            for bar_idx, bar in enumerate(ax_aggregate.patches):
                height = bar.get_height()
                ax_aggregate.text(
                    bar.get_x() + bar.get_width() / 2,
                    height / 1.1,
                    f'{height:.2f}',
                    ha='center',
                    va='center',
                    fontsize=9,
                    fontweight='bold',
                    color='black'
                )

        plt.show()


    # ------------------------------------------------------------------
    # Summary results plotting
    # ------------------------------------------------------------------
    def plot_summary_results(self):
        """
        Visualize how fairness and performance metrics evolve
        across iterations.

        - Line plots show metric trends over time by group
        - DPD and EOD are plotted separately (classification only)
        """

        print(f"\n{'='*150}")
        print(f"Summary of Metrics Over Time")
        print(f"{'='*150}")


        if not self.all_results:
            print("No results to plot. Run simulation first.")
            return
            
        
        records = []
        aggregate_records = []

        # Flatten results into records for plotting
        for it in self.all_results:
            for s, df in it["fairness"].items():
                for group in df.index:
                    for metric in df.columns:
                        records.append(
                            {
                                "iteration": it["iteration"],
                                "sensitive_var": s,
                                "group": group,
                                "metric": metric,
                                "value": df.loc[group, metric],
                            }
                        )
                
                # Collect DPD and EOD separately (classification only)
                if self.is_classification and s in it["aggregate_fairness"]:
                    aggregate_records.append({
                        "iteration": it["iteration"],
                        "sensitive_var": s,
                        "metric": "DPD",
                        "value": it["aggregate_fairness"][s]["DPD"],
                    })
                    aggregate_records.append({
                        "iteration": it["iteration"],
                        "sensitive_var": s,
                        "metric": "EOD",
                        "value": it["aggregate_fairness"][s]["EOD"],
                    })

        summary_df = pd.DataFrame(records)
        aggregate_df = pd.DataFrame(aggregate_records) if aggregate_records else None

        # Create separate plot for each sensitive variable
        for s in summary_df["sensitive_var"].unique():

            print(f"\n{'*'*90}")
            print(f"Sensitive Variable: {s}")
            print(f"{'*'*90}")
        
            sub = summary_df[summary_df["sensitive_var"] == s]
            


            g = sns.FacetGrid(
                sub,
                col="metric",
                hue="group",
                col_wrap=3,
                sharey=False,
                height=2.5,
                aspect=1.2,
                palette="Set1"
            )
            g.map(sns.lineplot, "iteration", "value", marker="o")
            g.add_legend(title="Group")
            g.figure.suptitle(
                f"Fairness Degradation Over Time – {s}", 
                y=1.02, 
                fontsize=14,
                fontweight='bold'
            )
            
            for ax in g.axes.flat:
                ax.grid(True, alpha=0.3)
                
            plt.show()
            
            # Plot DPD and EOD together in a separate figure (classification only)
            if aggregate_df is not None and not aggregate_df.empty:
                agg_sub = aggregate_df[aggregate_df["sensitive_var"] == s]
                
                if not agg_sub.empty:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    
                    sns.lineplot(
                        data=agg_sub,
                        x="iteration",
                        y="value",
                        hue="metric",
                        marker="o",
                        ax=ax,
                        palette={"DPD": "#e74c3c", "EOD": "#3498db"}
                    )
                    
                    ax.set_title(f"DPD & EOD Over Time – {s}", fontsize=12, fontweight='bold')
                    ax.set_xlabel("Iteration", fontsize=10)
                    ax.set_ylabel("Value", fontsize=10)
                    ax.legend(title="Metric", fontsize=8)
                    ax.grid(True, alpha=0.3)
                    
                    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                
                    plt.tight_layout()
                    plt.show()