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
    """Simulate how model performance and fairness change over time due to ML feedback loops.

    This simulator demonstrates how predictions used as future training labels can cause
    model performance and fairness metrics to degrade over successive iterations.

    Supports:
        - Classification: performance and fairness metrics by group
        - Regression: error and residual-based disparity metrics

    Parameters
    ----------
    model : estimator
        An sklearn-compatible model with fit() and predict() methods.
    dataset : pandas.DataFrame
        DataFrame containing features, target, and sensitive attribute columns.
    target_variable : str
        Name of the target column in the dataset.
    sensitive_variables : list of str
        Names of sensitive attribute columns for fairness analysis.
    display_func : callable, optional
        Function to display DataFrames. Default is print().

    Methods
    -------
    `set_show_iteration_data(show)`
        Toggle iteration data display.
    `run_simulation(n_iterations)`
        Execute the simulation.
    """

    def __init__(self, model, dataset, target_variable, sensitive_variables, display_func=None):
        self._base_model = model
        self._data = dataset.reset_index(drop=True).copy()
        self._target = target_variable
        self._sensitive_vars = sensitive_variables
        self._sensitive_data_original = self._data[self._sensitive_vars].copy()
        self._display_func = display_func if display_func else print
        self._show_iteration_data = True
        self._preprocess_data()
        self._is_classification = self._detect_task_type()
        self._all_results = []


    # ------------------------------------------------------------------
    # Setter for showing iteration data
    # ------------------------------------------------------------------
    def set_show_iteration_data(self, show):
        """Set whether to display data and plots for each iteration.

        Useful with higher number of iterations to reduce output length.

        Parameters
        ----------
        show : bool
            True to display iteration data and plots, False to hide them
            and show only the final summary.
        """
        self._show_iteration_data = show


    # ------------------------------------------------------------------
    # Preprocessing data
    # ------------------------------------------------------------------
    def _preprocess_data(self):
        """Prepare the dataset for modeling and fairness analysis.

        Performs the following preprocessing steps:
            1. Encode the target variable if it is categorical
            2. One-hot encode non-sensitive categorical features
            3. Encode sensitive variables for modeling while keeping
               originals for fairness evaluation
        """

        # Store for printing preprocessing steps
        self._target_encoder = None
        
        print(f"{'='*150}")
        print(f"Data Preprocessing")
        print(f"{'='*150}")
        print()

        # Encode target var if categorical
        if self._data[self._target].dtype == 'object' or isinstance(self._data[self._target].iloc[0], str):
            self._target_encoder = LabelEncoder()
            self._data[self._target] = self._target_encoder.fit_transform(self._data[self._target])
            print(f"Target variable '{self._target}' encoded: {dict(zip(self._target_encoder.classes_, self._target_encoder.transform(self._target_encoder.classes_)))}")
        
        # Identify categorical variables without target and sensitive variables
        categorical_cols = []
        for col in self._data.columns:
            if col != self._target and col not in self._sensitive_vars:
                if self._data[col].dtype == 'object' or self._data[col].dtype.name == 'category':
                    categorical_cols.append(col)
        
        # One-hot encode categorical features
        if categorical_cols:
            print(f"One-hot encoding categorical features: {categorical_cols}")
            self._data = pd.get_dummies(
                self._data, 
                columns=categorical_cols, 
                drop_first=True,
                dtype=int
            )
        
        # Encode sensitive variables for use as features (keep originals for fairness metrics)
        for s_var in self._sensitive_vars:
            if self._data[s_var].dtype == 'object' or isinstance(self._data[s_var].iloc[0], str):
                encoder = LabelEncoder()
                self._data[f"{s_var}_encoded"] = encoder.fit_transform(self._data[s_var])
                print(f"Sensitive variable '{s_var}' encoded for features: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")
        
        print(f"Dataset shape after preprocessing: {self._data.shape}")


    # ------------------------------------------------------------------
    # Detect model task type: Classification vs Regression
    # ------------------------------------------------------------------
    def _detect_task_type(self):
        """Detect whether the task is classification or regression.

        Determines the task type based on the target variable characteristics
        using sklearn's type_of_target utility.

        Returns
        -------
        bool
            True if classification (binary or multiclass), False if regression.
        """
        y = self._data[self._target]
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
    def _data_split(self, n_iterations):
        """Split the dataset into sequential chunks for simulation.

        The data is divided into (n_iterations + 1) parts so that
        each iteration trains on one part and tests on the next.

        Parameters
        ----------
        n_iterations : int
            Number of simulation iterations to run.

        Returns
        -------
        list of pandas.DataFrame
            Sequential data partitions for train/test splits.
        """
        n_parts = n_iterations + 1
        return np.array_split(self._data, n_parts)


    # ------------------------------------------------------------------
    # Calculate performance and fairness metrics for iteration
    # ------------------------------------------------------------------
    def _compute_metrics_by_group(self, y_true, y_pred, sensitive_series):
        """Compute performance and fairness metrics for each sensitive group.

        Calculates group-specific metrics using Fairlearn's MetricFrame:
            - Classification: Accuracy, F1, Precision, Recall,
              Selection Rate, TPR, FNR (binary only)
            - Regression: MAE, MSE, R², mean and std of residuals

        Parameters
        ----------
        y_true : array-like
            True target values.
        y_pred : array-like
            Predicted values from the model.
        sensitive_series : pandas.Series
            Sensitive attribute values for grouping.

        Returns
        -------
        pandas.DataFrame
            Metrics computed for each sensitive group.
        """
        if self._is_classification:
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
        """Compute aggregate fairness metrics for binary classification.

        Calculates Demographic Parity Difference (DPD) and Equalized Odds
        Difference (EOD) using Fairlearn metrics.

        Parameters
        ----------
        y_true : array-like
            True target values.
        y_pred : array-like
            Predicted values from the model.
        sensitive_series : pandas.Series
            Sensitive attribute values for fairness computation.

        Returns
        -------
        tuple of (float, float) or (None, None)
            (DPD, EOD) values for binary classification,
            or (None, None) if not a binary classification task.
        """
        n_classes = len(np.unique(y_true))
        is_binary = n_classes == 2
        
        if not self._is_classification or not is_binary:
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
        """Run the feedback-loop fairness degradation simulation.

        Executes the simulation by iteratively training models where predictions
        from previous iterations become training labels for subsequent ones,
        demonstrating how feedback loops affect fairness and performance.

        For each iteration:
            1. Split data sequentially
            2. Train the model on current data
            3. Replace future training labels with past predictions
            4. Evaluate on the next data split
            5. Compute and visualize group-level fairness metrics
            6. Store results for summary visualization

        Parameters
        ----------
        n_iterations : int
            Number of feedback loop iterations to simulate.
        """
        self._all_results = []
        partitions = self._data_split(n_iterations)

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
                train_df["original_target"] = train_df[self._target].copy()
                # Replace target with previous predictions
                train_df[self._target] = prev_predictions

            # Prepare features (remove target, original target, original sensitive vars - encoded versions remain)
            X_train = (
                train_df
                .drop(columns=[self._target, "original_target"] + self._sensitive_vars, errors="ignore")
                .select_dtypes(include=[np.number])
            )

            y_train = train_df[self._target]  # Now uses predictions as target

            X_test = (
                test_df
                .drop(columns=[self._target] + self._sensitive_vars, errors="ignore")
                .select_dtypes(include=[np.number])
            )

            y_test = test_df[self._target]

            # train a new model clone
            model = clone(self._base_model)
            model.fit(X_train, y_train)

            # Predict and store predictions for next iteration
            y_pred = model.predict(X_test)
            prev_predictions = y_pred

            if self._show_iteration_data:
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
            for s in self._sensitive_vars:
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

                if self._show_iteration_data:
                    print(f"\nSensitive variable: {s} (Iteration {i+1}/{n_iterations})")
                    
                    # Display group metrics table
                    self._display_func(grouped_metrics)
                    print()
                    
                    # Display DPD and EOD table (binary classification only)
                    if demo_parity is not None and eqo is not None:
                        aggregate_df = pd.DataFrame({
                            "DPD": [demo_parity],
                            "EOD": [eqo]
                        }, index=[s])
                        self._display_func(aggregate_df)

                    self._plot_iteration_results(grouped_metrics, s, i + 1, demo_parity, eqo)

            self._all_results.append(iteration_result)

        # After all iterations, show summary over time
        self._plot_summary_results()


    # ------------------------------------------------------------------
    # Iteration results plotting
    # ------------------------------------------------------------------
    def _plot_iteration_results(self, df, sensitive_var, iteration, dpd=None, eod=None):
        """Visualize fairness metrics for a single iteration.

        Creates bar charts for each metric by sensitive group, with optional
        display of aggregate fairness metrics (DPD and EOD) for binary classification.

        Parameters
        ----------
        df : pandas.DataFrame
            Metrics DataFrame with groups as index and metrics as columns.
        sensitive_var : str
            Name of the sensitive variable being analyzed.
        iteration : int
            Current iteration number for the plot title.
        dpd : float, optional
            Demographic Parity Difference value. Default is None.
        eod : float, optional
            Equalized Odds Difference value. Default is None.
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
    def _plot_summary_results(self):
        """Visualize how fairness and performance metrics evolve across iterations.

        Creates line plots showing metric trends over time by group,
        with separate plots for DPD and EOD (classification only).
        """

        print(f"\n{'='*150}")
        print(f"Summary of Metrics Over Time")
        print(f"{'='*150}")


        if not self._all_results:
            print("No results to plot. Run simulation first.")
            return
            
        records = []
        aggregate_records = []

        # Flatten results into records for plotting
        for it in self._all_results:
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
                if self._is_classification and s in it["aggregate_fairness"]:
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