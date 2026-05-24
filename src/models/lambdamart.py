"""
LambdaMART ranking model.
Uses XGBoost with rank:ndcg objective (LambdaRank loss).
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, List, Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)


class LambdaMARTRanker:
    """
    LambdaMART learning-to-rank model using XGBoost.
    Optimizes NDCG directly via LambdaRank loss.
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Args:
            config: Model hyperparameters dict. Uses defaults if None.
        """
        self.config = config or {}
        self.model = None
        self.best_iteration = None
        self.feature_names = None
        self.evals_result = {}
        self._fitted = False

        # Extract hyperparams from config
        self.params = {
            "objective": self.config.get("objective", "rank:ndcg"),
            "learning_rate": self.config.get("learning_rate", 0.05),
            "n_estimators": self.config.get("n_estimators", 1000),
            "max_depth": self.config.get("max_depth", 6),
            "subsample": self.config.get("subsample", 0.8),
            "colsample_bytree": self.config.get("colsample_bytree", 0.8),
            "min_child_weight": self.config.get("min_child_weight", 10),
            "gamma": self.config.get("gamma", 0.1),
            "reg_alpha": self.config.get("reg_alpha", 0.1),
            "reg_lambda": self.config.get("reg_lambda", 1.0),
            "tree_method": self.config.get("tree_method", "hist"),
            "random_state": self.config.get("random_state", 42),
        }
        self.early_stopping_rounds = self.config.get("early_stopping_rounds", 50)
        self.eval_metrics = self.config.get("eval_metric", ["ndcg@5", "ndcg@10"])

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        qid_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        qid_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "LambdaMARTRanker":
        """
        Train LambdaMART model.

        Args:
            X_train: Training feature matrix
            y_train: Training relevance labels (0-4)
            qid_train: Training query IDs
            X_val: Validation features (optional, for early stopping)
            y_val: Validation labels
            qid_val: Validation query IDs
            feature_names: Feature column names

        Returns:
            self
        """
        logger.info("=" * 60)
        logger.info("Training LambdaMART model")
        logger.info("=" * 60)
        logger.info(f"  Training samples: {len(X_train):,}")
        logger.info(f"  Features: {X_train.shape[1]}")
        logger.info(f"  Hyperparams: {self.params}")

        self.feature_names = feature_names

        # Initialize XGBRanker
        model_params = {k: v for k, v in self.params.items() if k != "n_estimators"}
        n_est = self.params.get("n_estimators", 1000)

        # Determine if we have validation data for early stopping
        has_val = X_val is not None and y_val is not None and qid_val is not None

        init_kwargs = {
            **model_params,
            "n_estimators": n_est,
            "eval_metric": self.eval_metrics,
        }
        if has_val:
            init_kwargs["early_stopping_rounds"] = self.early_stopping_rounds

        self.model = xgb.XGBRanker(**init_kwargs)

        # Prepare fit kwargs
        fit_kwargs = {
            "X": X_train,
            "y": y_train,
            "qid": qid_train,
            "verbose": 50,
        }

        # Add validation set if provided
        if has_val:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["eval_qid"] = [qid_val]

        # Train
        self.model.fit(**fit_kwargs)

        self.best_iteration = self.model.best_iteration if hasattr(self.model, 'best_iteration') else None
        self._fitted = True

        logger.info(f"\nTraining complete.")
        if self.best_iteration is not None:
            logger.info(f"  Best iteration: {self.best_iteration}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict relevance scores for documents.

        Args:
            X: Feature matrix

        Returns:
            Array of predicted relevance scores
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def get_feature_importance(
        self,
        importance_type: str = "gain",
    ) -> Dict[str, float]:
        """
        Get feature importance scores.

        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')

        Returns:
            Dict mapping feature names to importance scores
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted.")

        importance = self.model.get_booster().get_score(
            importance_type=importance_type
        )

        # Map feature indices to names
        if self.feature_names:
            named_importance = {}
            for key, value in importance.items():
                # XGBoost uses 'f0', 'f1', etc. as feature names
                if key.startswith("f"):
                    try:
                        idx = int(key[1:])
                        if idx < len(self.feature_names):
                            named_importance[self.feature_names[idx]] = value
                        else:
                            named_importance[key] = value
                    except ValueError:
                        named_importance[key] = value
                else:
                    named_importance[key] = value
            importance = named_importance

        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        return importance

    def save(self, path: str) -> None:
        """Save model to file."""
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> "LambdaMARTRanker":
        """Load model from file."""
        self.model = xgb.XGBRanker()
        self.model.load_model(path)
        self._fitted = True
        logger.info(f"Model loaded from {path}")
        return self
