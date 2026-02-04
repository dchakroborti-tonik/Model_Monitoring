"""
Model Monitoring System - OOP Architecture

This module provides a comprehensive object-oriented architecture for monitoring
machine learning models with PSI, CSI, and Gini calculations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score

# ============================================================================
# Data Models
# ============================================================================


@dataclass
class MonitoringConfig:
    """Configuration for monitoring calculations."""

    bins: int = 10
    score_column: str = "score"
    target_column: str = "target"
    feature_columns: Optional[List[str]] = None


@dataclass
class MonitoringResult:
    """Container for monitoring results."""

    psi_data: pd.DataFrame
    csi_data: pd.DataFrame
    gini_score: Optional[float] = None
    metadata: Optional[Dict] = None


@dataclass
class VisualizationResult:
    """Container for visualization outputs."""

    psi_plot: plt.Figure
    csi_heatmap: plt.Figure
    gini_plot: Optional[plt.Figure] = None


# ============================================================================
# Abstract Base Classes
# ============================================================================


class IMetricCalculator(ABC):
    """Interface for metric calculation strategies."""

    @abstractmethod
    def calculate(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, config: MonitoringConfig
    ) -> pd.DataFrame:
        """Calculate the metric."""
        pass


class IVisualizer(ABC):
    """Interface for visualization strategies."""

    @abstractmethod
    def visualize(self, data: pd.DataFrame, **kwargs) -> plt.Figure:
        """Create visualization from data."""
        pass


class IDataProcessor(ABC):
    """Interface for data processing operations."""

    @abstractmethod
    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Process input data."""
        pass


# ============================================================================
# Metric Calculators
# ============================================================================


class PSICalculator(IMetricCalculator):
    """Population Stability Index calculator."""

    def calculate(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, config: MonitoringConfig
    ) -> pd.DataFrame:
        """
        Calculate PSI for the score column.

        Args:
            train_df: Training/baseline dataset
            test_df: Test/monitoring dataset
            config: Configuration with bins and score column

        Returns:
            DataFrame with PSI calculations per bin
        """
        score_col = config.score_column
        bins = config.bins

        # Create bins based on training data
        train_scores = train_df[score_col]
        bin_edges = pd.qcut(train_scores, q=bins, retbins=True, duplicates="drop")[1]

        # Calculate distributions
        train_binned = pd.cut(train_scores, bins=bin_edges, include_lowest=True)
        test_binned = pd.cut(test_df[score_col], bins=bin_edges, include_lowest=True)

        train_dist = train_binned.value_counts(normalize=True).sort_index()
        test_dist = test_binned.value_counts(normalize=True).sort_index()

        # Calculate PSI
        psi_values = (test_dist - train_dist) * pd.np.log(test_dist / train_dist)

        return pd.DataFrame(
            {
                "bin": train_dist.index.astype(str),
                "train_pct": train_dist.values,
                "test_pct": test_dist.values,
                "psi": psi_values.values,
                "total_psi": psi_values.sum(),
            }
        )


class CSICalculator(IMetricCalculator):
    """Characteristic Stability Index calculator."""

    def calculate(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, config: MonitoringConfig
    ) -> pd.DataFrame:
        """
        Calculate CSI for all feature columns.

        Args:
            train_df: Training/baseline dataset
            test_df: Test/monitoring dataset
            config: Configuration with bins and feature columns

        Returns:
            DataFrame with CSI calculations per feature
        """
        feature_cols = config.feature_columns or []
        bins = config.bins

        csi_results = []

        for feature in feature_cols:
            try:
                # Handle different bin specifications
                feature_bins = bins if isinstance(bins, int) else bins.get(feature, 10)

                # Create bins based on training data
                train_values = train_df[feature].dropna()

                if pd.api.types.is_numeric_dtype(train_values):
                    bin_edges = pd.qcut(
                        train_values, q=feature_bins, retbins=True, duplicates="drop"
                    )[1]
                    train_binned = pd.cut(
                        train_values, bins=bin_edges, include_lowest=True
                    )
                    test_binned = pd.cut(
                        test_df[feature].dropna(), bins=bin_edges, include_lowest=True
                    )
                else:
                    # Categorical feature
                    train_binned = train_df[feature]
                    test_binned = test_df[feature]

                # Calculate distributions
                train_dist = train_binned.value_counts(normalize=True)
                test_dist = test_binned.value_counts(normalize=True)

                # Align distributions
                all_bins = train_dist.index.union(test_dist.index)
                train_dist = train_dist.reindex(all_bins, fill_value=0.0001)
                test_dist = test_dist.reindex(all_bins, fill_value=0.0001)

                # Calculate CSI
                csi_value = (
                    (test_dist - train_dist) * pd.np.log(test_dist / train_dist)
                ).sum()

                csi_results.append(
                    {
                        "feature": feature,
                        "csi": csi_value,
                        "status": self._get_csi_status(csi_value),
                    }
                )

            except Exception as e:
                print(f"Error calculating CSI for {feature}: {str(e)}")
                csi_results.append({"feature": feature, "csi": None, "status": "error"})

        return pd.DataFrame(csi_results)

    @staticmethod
    def _get_csi_status(csi_value: float) -> str:
        """Determine CSI status based on threshold."""
        if csi_value < 0.1:
            return "stable"
        elif csi_value < 0.25:
            return "warning"
        else:
            return "critical"


class GiniCalculator:
    """Gini coefficient calculator."""

    @staticmethod
    def calculate(y_true: pd.Series, y_score: pd.Series) -> float:
        """
        Calculate Gini coefficient from true labels and predicted scores.

        Args:
            y_true: True binary labels
            y_score: Predicted scores/probabilities

        Returns:
            Gini coefficient (0-1)
        """
        auc = roc_auc_score(y_true, y_score)
        return 2 * auc - 1


# ============================================================================
# Visualizers
# ============================================================================


class PSIVisualizer(IVisualizer):
    """Visualizer for PSI distributions."""

    def visualize(
        self, data: pd.DataFrame, month: str = None, score_col: str = "score"
    ) -> plt.Figure:
        """
        Create PSI distribution plot.

        Args:
            data: PSI DataFrame from PSICalculator
            month: Optional month label for title
            score_col: Name of score column for labeling

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        x = range(len(data))
        width = 0.35

        ax.bar(
            [i - width / 2 for i in x],
            data["train_pct"],
            width,
            label="Train",
            alpha=0.8,
            color="skyblue",
        )
        ax.bar(
            [i + width / 2 for i in x],
            data["test_pct"],
            width,
            label="Test",
            alpha=0.8,
            color="coral",
        )

        ax.set_xlabel("Bins")
        ax.set_ylabel("Proportion")
        ax.set_title(
            f"PSI Distribution - {score_col}" + (f" ({month})" if month else "")
        )
        ax.set_xticks(x)
        ax.set_xticklabels(data["bin"], rotation=45, ha="right")
        ax.legend()

        # Add PSI value as text
        total_psi = data["total_psi"].iloc[0]
        ax.text(
            0.02,
            0.98,
            f"Total PSI: {total_psi:.4f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        return fig


class CSIHeatmapVisualizer(IVisualizer):
    """Visualizer for CSI heatmap."""

    def visualize(self, data: pd.DataFrame, **kwargs) -> plt.Figure:
        """
        Create CSI heatmap visualization.

        Args:
            data: CSI DataFrame from CSICalculator

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, max(6, len(data) * 0.4)))

        # Prepare data for heatmap
        csi_values = data.set_index("feature")["csi"].to_frame()

        # Create color map based on CSI thresholds
        colors = csi_values["csi"].apply(
            lambda x: "green" if x < 0.1 else "yellow" if x < 0.25 else "red"
        )

        # Create heatmap
        im = ax.imshow(
            csi_values.values.reshape(-1, 1),
            cmap="RdYlGn_r",
            aspect="auto",
            vmin=0,
            vmax=0.5,
        )

        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data["feature"])
        ax.set_xticks([0])
        ax.set_xticklabels(["CSI"])
        ax.set_title("Feature Stability (CSI)")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("CSI Value", rotation=270, labelpad=20)

        # Add text annotations
        for i, (idx, row) in enumerate(data.iterrows()):
            text_color = "white" if row["csi"] > 0.25 else "black"
            ax.text(
                0,
                i,
                f"{row['csi']:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontweight="bold",
            )

        plt.tight_layout()
        return fig


class GiniVisualizer(IVisualizer):
    """Visualizer for Gini coefficient."""

    def visualize(self, data: float, **kwargs) -> plt.Figure:
        """
        Create Gini coefficient bar plot.

        Args:
            data: Gini coefficient value

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.bar(["Gini"], [data], color="skyblue", width=0.5)
        ax.set_title("Gini Coefficient")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="Baseline")
        ax.legend()

        # Add value text
        ax.text(0, data + 0.05, f"{data:.4f}", ha="center", fontweight="bold")

        plt.tight_layout()
        return fig


# ============================================================================
# Data Processors
# ============================================================================


class ModelRunDataProcessor(IDataProcessor):
    """Processor for flattening ML model run details."""

    def process(
        self,
        data: pd.DataFrame,
        feature_keys: List[str] = None,
        prediction_keys: List[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Flatten nested ML model run details into a flat DataFrame.

        Args:
            data: Input DataFrame with nested structures
            feature_keys: Keys to extract from feature dictionaries
            prediction_keys: Keys to extract from prediction dictionaries

        Returns:
            Flattened DataFrame
        """
        # Placeholder implementation - would depend on actual data structure
        flattened_data = data.copy()

        # Example: Extract nested features
        if feature_keys and "features" in data.columns:
            for key in feature_keys:
                flattened_data[f"feature_{key}"] = data["features"].apply(
                    lambda x: x.get(key) if isinstance(x, dict) else None
                )

        # Example: Extract nested predictions
        if prediction_keys and "predictions" in data.columns:
            for key in prediction_keys:
                flattened_data[f"pred_{key}"] = data["predictions"].apply(
                    lambda x: x.get(key) if isinstance(x, dict) else None
                )

        return flattened_data


# ============================================================================
# Service Layer
# ============================================================================


class MonitoringService:
    """Service for coordinating monitoring calculations."""

    def __init__(
        self,
        psi_calculator: PSICalculator = None,
        csi_calculator: CSICalculator = None,
        gini_calculator: GiniCalculator = None,
    ):
        self.psi_calculator = psi_calculator or PSICalculator()
        self.csi_calculator = csi_calculator or CSICalculator()
        self.gini_calculator = gini_calculator or GiniCalculator()

    def calculate_all_metrics(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, config: MonitoringConfig
    ) -> MonitoringResult:
        """
        Calculate all monitoring metrics.

        Args:
            train_df: Training/baseline dataset
            test_df: Test/monitoring dataset
            config: Monitoring configuration

        Returns:
            MonitoringResult with all calculated metrics
        """
        psi_data = self.psi_calculator.calculate(train_df, test_df, config)
        csi_data = self.csi_calculator.calculate(train_df, test_df, config)

        gini_score = None
        if config.target_column in test_df.columns:
            gini_score = self.gini_calculator.calculate(
                test_df[config.target_column], test_df[config.score_column]
            )

        return MonitoringResult(
            psi_data=psi_data,
            csi_data=csi_data,
            gini_score=gini_score,
            metadata={"config": config},
        )


class VisualizationService:
    """Service for coordinating visualizations."""

    def __init__(
        self,
        psi_visualizer: PSIVisualizer = None,
        csi_visualizer: CSIHeatmapVisualizer = None,
        gini_visualizer: GiniVisualizer = None,
    ):
        self.psi_visualizer = psi_visualizer or PSIVisualizer()
        self.csi_visualizer = csi_visualizer or CSIHeatmapVisualizer()
        self.gini_visualizer = gini_visualizer or GiniVisualizer()

    def create_all_visualizations(
        self, monitoring_result: MonitoringResult, month: str = None
    ) -> VisualizationResult:
        """
        Create all visualizations from monitoring results.

        Args:
            monitoring_result: Results from MonitoringService
            month: Optional month label

        Returns:
            VisualizationResult with all plots
        """
        psi_plot = self.psi_visualizer.visualize(
            monitoring_result.psi_data,
            month=month,
            score_col=monitoring_result.metadata["config"].score_column,
        )

        csi_heatmap = self.csi_visualizer.visualize(monitoring_result.csi_data)

        gini_plot = None
        if monitoring_result.gini_score is not None:
            gini_plot = self.gini_visualizer.visualize(monitoring_result.gini_score)

        return VisualizationResult(
            psi_plot=psi_plot, csi_heatmap=csi_heatmap, gini_plot=gini_plot
        )


# ============================================================================
# Facade/Manager Layer
# ============================================================================


class ModelMonitoringManager:
    """
    High-level facade for model monitoring operations.

    This class provides a simplified interface for the monitoring system,
    coordinating between services and providing convenient methods.
    """

    def __init__(
        self,
        monitoring_service: MonitoringService = None,
        visualization_service: VisualizationService = None,
        data_processor: ModelRunDataProcessor = None,
    ):
        self.monitoring_service = monitoring_service or MonitoringService()
        self.visualization_service = visualization_service or VisualizationService()
        self.data_processor = data_processor or ModelRunDataProcessor()

    def monitor_model(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        score_col: str,
        feature_cols: List[str],
        bins: Union[int, Dict] = 10,
        target_col: str = None,
        month: str = None,
    ) -> Dict:
        """
        Comprehensive model monitoring with PSI, CSI, and visualizations.

        Args:
            train_df: Training/baseline dataset
            test_df: Test/monitoring dataset
            score_col: Name of the score/prediction column
            feature_cols: List of feature columns to monitor
            bins: Number of bins or dict mapping features to bin counts
            target_col: Optional target column for Gini calculation
            month: Optional month label for visualizations

        Returns:
            Dictionary containing metrics and visualizations
        """
        # Create configuration
        config = MonitoringConfig(
            bins=bins,
            score_column=score_col,
            target_column=target_col,
            feature_columns=feature_cols,
        )

        # Calculate metrics
        monitoring_result = self.monitoring_service.calculate_all_metrics(
            train_df, test_df, config
        )

        # Create visualizations
        viz_result = self.visualization_service.create_all_visualizations(
            monitoring_result, month=month
        )

        # Return comprehensive results
        return {
            "psi": monitoring_result.psi_data,
            "csi": monitoring_result.csi_data,
            "gini": monitoring_result.gini_score,
            "psi_plot": viz_result.psi_plot,
            "csi_heatmap": viz_result.csi_heatmap,
            "gini_plot": viz_result.gini_plot,
        }

    def flatten_model_run_details(
        self, table_name: str, feature_keys: List[str], prediction_keys: List[str]
    ) -> pd.DataFrame:
        """
        Flatten ML model run details from a table.

        Args:
            table_name: Name of the source table
            feature_keys: Keys to extract from features
            prediction_keys: Keys to extract from predictions

        Returns:
            Flattened DataFrame
        """
        # Load data (placeholder - would load from actual source)
        data = pd.DataFrame()  # Load from table_name

        return self.data_processor.process(
            data, feature_keys=feature_keys, prediction_keys=prediction_keys
        )


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Create sample data
    np.random.seed(42)
    train_df = pd.DataFrame(
        {
            "score": np.random.beta(2, 5, 1000),
            "feature1": np.random.normal(0, 1, 1000),
            "feature2": np.random.uniform(0, 10, 1000),
            "target": np.random.binomial(1, 0.3, 1000),
        }
    )

    test_df = pd.DataFrame(
        {
            "score": np.random.beta(2.2, 5, 500),
            "feature1": np.random.normal(0.2, 1.1, 500),
            "feature2": np.random.uniform(0, 11, 500),
            "target": np.random.binomial(1, 0.3, 500),
        }
    )

    # Initialize manager
    manager = ModelMonitoringManager()

    # Run monitoring
    results = manager.monitor_model(
        train_df=train_df,
        test_df=test_df,
        score_col="score",
        feature_cols=["feature1", "feature2"],
        bins=10,
        target_col="target",
        month="2025-10",
    )

    print("PSI Results:")
    print(results["psi"])
    print("\nCSI Results:")
    print(results["csi"])
    print(f"\nGini Score: {results['gini']:.4f}")
