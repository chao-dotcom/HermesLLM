"""
Comet ML integration for experiment tracking during training.

This module provides utilities for tracking training experiments with Comet ML,
including metrics, hyperparameters, models, and artifacts.
"""

import os
from typing import Any, Dict, Optional

from loguru import logger

try:
    import comet_ml
    COMET_AVAILABLE = True
except ImportError:
    logger.warning("Comet ML not installed. Install with: pip install comet-ml")
    COMET_AVAILABLE = False
    comet_ml = None


class CometTracker:
    """Comet ML experiment tracker for training runs."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        project_name: Optional[str] = None,
        workspace: Optional[str] = None,
        experiment_name: Optional[str] = None,
        auto_log_parameters: bool = True,
        auto_log_metrics: bool = True,
    ):
        """
        Initialize Comet ML tracker.
        
        Args:
            api_key: Comet ML API key
            project_name: Project name
            workspace: Workspace name
            experiment_name: Experiment name
            auto_log_parameters: Auto-log hyperparameters
            auto_log_metrics: Auto-log metrics
        """
        self.experiment = None
        
        if not COMET_AVAILABLE:
            logger.warning("Comet ML not available. Tracking disabled.")
            return
        
        # Get settings
        from hermes.config import get_settings
        settings = get_settings()
        
        api_key = api_key or settings.comet_api_key or os.getenv("COMET_API_KEY")
        project_name = project_name or settings.comet_project or os.getenv("COMET_PROJECT")
        workspace = workspace or settings.comet_workspace or os.getenv("COMET_WORKSPACE")
        
        if not api_key:
            logger.warning("COMET_API_KEY not set. Comet ML tracking disabled.")
            return
        
        try:
            self.experiment = comet_ml.Experiment(
                api_key=api_key,
                project_name=project_name,
                workspace=workspace,
                auto_log_parameters=auto_log_parameters,
                auto_log_metrics=auto_log_metrics,
            )
            
            if experiment_name:
                self.experiment.set_name(experiment_name)
            
            logger.info(f"✅ Comet ML experiment initialized: {project_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Comet ML experiment: {e}")
            self.experiment = None
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters.
        
        Args:
            params: Dictionary of parameters
        """
        if self.experiment:
            try:
                self.experiment.log_parameters(params)
            except Exception as e:
                logger.debug(f"Failed to log parameters: {e}")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> None:
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Training step
            epoch: Training epoch
        """
        if self.experiment:
            try:
                self.experiment.log_metrics(metrics, step=step, epoch=epoch)
            except Exception as e:
                logger.debug(f"Failed to log metrics: {e}")
    
    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> None:
        """
        Log a single metric.
        
        Args:
            name: Metric name
            value: Metric value
            step: Training step
            epoch: Training epoch
        """
        if self.experiment:
            try:
                self.experiment.log_metric(name, value, step=step, epoch=epoch)
            except Exception as e:
                logger.debug(f"Failed to log metric: {e}")
    
    def log_model(
        self,
        model_name: str,
        file_or_folder: str,
        overwrite: bool = False,
    ) -> None:
        """
        Log model checkpoint.
        
        Args:
            model_name: Name for the model
            file_or_folder: Path to model file or folder
            overwrite: Whether to overwrite existing model
        """
        if self.experiment:
            try:
                self.experiment.log_model(
                    model_name,
                    file_or_folder,
                    overwrite=overwrite,
                )
                logger.info(f"Logged model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to log model: {e}")
    
    def log_artifact(
        self,
        artifact: str,
        artifact_type: Optional[str] = None,
    ) -> None:
        """
        Log artifact (file, dataset, etc.).
        
        Args:
            artifact: Path to artifact
            artifact_type: Type of artifact
        """
        if self.experiment:
            try:
                self.experiment.log_artifact(artifact, artifact_type=artifact_type)
            except Exception as e:
                logger.debug(f"Failed to log artifact: {e}")
    
    def log_dataset_hash(self, dataset_hash: str) -> None:
        """
        Log dataset hash/identifier.
        
        Args:
            dataset_hash: Dataset hash or identifier
        """
        if self.experiment:
            try:
                self.experiment.log_other("dataset_hash", dataset_hash)
            except Exception as e:
                logger.debug(f"Failed to log dataset hash: {e}")
    
    def add_tags(self, tags: list[str]) -> None:
        """
        Add tags to experiment.
        
        Args:
            tags: List of tags
        """
        if self.experiment:
            try:
                self.experiment.add_tags(tags)
            except Exception as e:
                logger.debug(f"Failed to add tags: {e}")
    
    def log_code(self, code: Optional[str] = None) -> None:
        """
        Log code/script.
        
        Args:
            code: Code to log (logs current script if None)
        """
        if self.experiment:
            try:
                if code:
                    self.experiment.log_code(code=code)
                else:
                    self.experiment.log_code()
            except Exception as e:
                logger.debug(f"Failed to log code: {e}")
    
    def log_confusion_matrix(
        self,
        y_true,
        y_predicted,
        labels: Optional[list] = None,
    ) -> None:
        """
        Log confusion matrix.
        
        Args:
            y_true: True labels
            y_predicted: Predicted labels
            labels: Label names
        """
        if self.experiment:
            try:
                self.experiment.log_confusion_matrix(
                    y_true,
                    y_predicted,
                    labels=labels,
                )
            except Exception as e:
                logger.debug(f"Failed to log confusion matrix: {e}")
    
    def end(self) -> None:
        """End the experiment."""
        if self.experiment:
            try:
                self.experiment.end()
                logger.info("Comet ML experiment ended")
            except Exception as e:
                logger.debug(f"Failed to end experiment: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end()


def create_comet_experiment(
    experiment_name: str,
    tags: Optional[list[str]] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> CometTracker:
    """
    Create a Comet ML experiment.
    
    Args:
        experiment_name: Name for the experiment
        tags: Tags to add
        parameters: Hyperparameters to log
        
    Returns:
        CometTracker instance
        
    Example:
        with create_comet_experiment("training_run", tags=["LoRA", "SFT"]) as tracker:
            tracker.log_parameters({"lr": 3e-4, "epochs": 3})
            for epoch in range(3):
                loss = train_epoch()
                tracker.log_metric("loss", loss, epoch=epoch)
    """
    tracker = CometTracker(experiment_name=experiment_name)
    
    if tags:
        tracker.add_tags(tags)
    
    if parameters:
        tracker.log_parameters(parameters)
    
    return tracker
