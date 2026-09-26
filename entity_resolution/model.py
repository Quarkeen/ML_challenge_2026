"""
Platform-Agnostic Pairwise Scoring Model.
Supports XGBoost and LightGBM with automatic CUDA / NVIDIA GPU acceleration
(including RTX 4500 Ada generation) and seamless CPU fallback.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class PairwiseRanker:
    """
    Supervised pairwise ranking & classification model.
    Auto-detects GPU (e.g. RTX 4500 Ada) and handles feature extraction & probability scoring.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model_type = self.config.get("model_type", "xgboost").lower()
        self.device = self._resolve_device(self.config.get("device", "auto"))
        self.model = None
        self.feature_names: List[str] = []

    def _resolve_device(self, preferred: str) -> str:
        """Resolve device to 'cuda' or 'cpu' safely."""
        if preferred in ("cuda", "gpu"):
            return "cuda"
        if preferred == "cpu":
            return "cpu"

        # Auto-detect
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass

        # Check xgboost cuda support directly
        try:
            import xgboost as xgb
            test_mat = xgb.DMatrix(np.zeros((2, 2)), label=[0, 1])
            xgb.train({"tree_method": "hist", "device": "cuda"}, test_mat, num_boost_round=1)
            return "cuda"
        except Exception:
            pass

        return "cpu"

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> "PairwiseRanker":
        """
        Train pairwise model with early stopping on validation set.
        """
        # Exclude ID columns if present
        ignore_cols = {"source1_entity_id", "candidate_id", "label", "target"}
        feat_cols = [c for c in X_train.columns if c not in ignore_cols]
        self.feature_names = feat_cols

        X_tr = X_train[feat_cols].astype(np.float32).fillna(0.0).values
        y_tr = np.asarray(y_train, dtype=np.float32)

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_v = X_val[feat_cols].astype(np.float32).fillna(0.0).values
            y_v = np.asarray(y_val, dtype=np.float32)

        if self.model_type == "xgboost":
            import xgboost as xgb
            
            xgb_params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": self.config.get("max_depth", 6),
                "learning_rate": self.config.get("learning_rate", 0.05),
                "subsample": self.config.get("subsample", 0.8),
                "colsample_bytree": self.config.get("colsample_bytree", 0.8),
                "seed": self.config.get("seed", 42),
                "tree_method": "hist",
                "device": self.device,
            }
            if self.device == "cpu":
                xgb_params["n_jobs"] = self.config.get("n_jobs", -1)

            dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_cols)
            evals = [(dtrain, "train")]
            if has_val:
                dval = xgb.DMatrix(X_v, label=y_v, feature_names=feat_cols)
                evals.append((dval, "val"))

            self.model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=self.config.get("n_estimators", 500),
                evals=evals,
                early_stopping_rounds=40 if has_val else None,
                verbose_eval=50 if has_val else False,
            )

        elif self.model_type == "lightgbm":
            import lightgbm as lgb
            
            lgb_params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "learning_rate": self.config.get("learning_rate", 0.05),
                "num_leaves": 2 ** self.config.get("max_depth", 6) - 1,
                "subsample": self.config.get("subsample", 0.8),
                "colsample_bytree": self.config.get("colsample_bytree", 0.8),
                "seed": self.config.get("seed", 42),
                "verbose": -1,
            }
            if self.device == "cuda":
                lgb_params["device_type"] = "cuda"
            else:
                lgb_params["n_jobs"] = self.config.get("n_jobs", -1)

            trn_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols)
            val_data = lgb.Dataset(X_v, label=y_v, feature_name=feat_cols, reference=trn_data) if has_val else None

            callbacks = [lgb.early_stopping(40, verbose=False), lgb.log_evaluation(50)] if has_val else []
            self.model = lgb.train(
                lgb_params,
                trn_data,
                num_boost_round=self.config.get("n_estimators", 500),
                valid_sets=[trn_data, val_data] if has_val else [trn_data],
                callbacks=callbacks,
            )

        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Compute matching probability scores for candidate pairs."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        if isinstance(X, np.ndarray):
            if X.ndim != 2 or X.shape[1] != len(self.feature_names):
                raise ValueError("Feature array must have one column per trained feature")
            X_mat = np.ascontiguousarray(X, dtype=np.float32)
            if np.isnan(X_mat).any():
                X_mat = np.nan_to_num(X_mat, nan=0.0, posinf=np.inf, neginf=-np.inf)
        else:
            X_mat = X[self.feature_names].astype(np.float32).fillna(0.0).values
        if len(X_mat) == 0:
            return np.array([])

        if self.model_type == "xgboost":
            import xgboost as xgb
            dmat = xgb.DMatrix(X_mat, feature_names=self.feature_names)
            return self.model.predict(dmat)
        elif self.model_type == "lightgbm":
            return self.model.predict(X_mat)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance dictionary."""
        if self.model is None:
            return {}
        if self.model_type == "xgboost":
            scores = self.model.get_score(importance_type="gain")
            return {k: float(scores.get(k, 0.0)) for k in self.feature_names}
        elif self.model_type == "lightgbm":
            importances = self.model.feature_importance(importance_type="gain")
            return dict(zip(self.feature_names, [float(x) for x in importances]))
        return {}

    def save(self, filepath: Union[str, Path]) -> None:
        """Save model and metadata to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "config": self.config,
        }
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if self.model_type == "xgboost":
            self.model.save_model(str(path))
        elif self.model_type == "lightgbm":
            self.model.save_model(str(path))

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "PairwiseRanker":
        """Load trained model and metadata from disk."""
        path = Path(filepath)
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        ranker = cls(config=meta["config"])
        ranker.model_type = meta["model_type"]
        ranker.feature_names = meta["feature_names"]

        if ranker.model_type == "xgboost":
            import xgboost as xgb
            ranker.model = xgb.Booster()
            ranker.model.load_model(str(path))
        elif ranker.model_type == "lightgbm":
            import lightgbm as lgb
            ranker.model = lgb.Booster(model_file=str(path))

        return ranker
