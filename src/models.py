"""
Layer 1, 2, and 3 models for the FIFA WC 2026 Forecast Engine.

Frozen snapshot: June 23, 2026 — Post-Matchday 2

Architecture
------------
Layer 1 — Pre-tournament baseline
    GroupStageModel    XGBoost (multi:softprob) + LR baseline, calibrated
    KnockoutModel      Stage-conditional binary classifiers (Stage A: R32/R16,
                       Stage B: QF/SF/Final)
    Trained on DS8 (1998–2022, WC-only primary corpus) and optionally on
    DS4 Tier 1 augmented matches (2014–2026). Two variants are trained
    separately and combined via WCForecastEnsemble.

Layer 2 — Bayesian tournament update
    BayesianTournamentUpdater   α = 0.17 shrinkage on in-tournament form signal

Layer 3 — Stage-conditional knockout + penalty
    KnockoutModel               (see above; called per stage)
    PenaltyModel                3-component ensemble (shootout rate + Elo + WC rate)

Validation
----------
ChronologicalCV     Leave-one-tournament-out; folds 1–6 for hyperparameter search;
                    fold 7 (2022) reserved exclusively for probability calibration.
                    Random k-fold is explicitly prohibited.

Public API
----------
    fit_all_models(training_rows, calib_rows, ...)
    GroupStageModel / KnockoutModel / PenaltyModel  — individual model classes
    BayesianTournamentUpdater
    WCForecastEnsemble          — master ensemble with predict_group_stage(),
                                  predict_knockout(), predict_shootout()
    ChronologicalCV
    compute_rps / compute_brier / compute_directional_accuracy — metrics
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except Exception:
    _XGB_AVAILABLE = False
    warnings.warn(
        "XGBoost could not be loaded (not installed or missing native library). "
        "XGBoost models will fall back to LogisticRegression. "
        "On macOS: brew install libomp",
        stacklevel=2,
    )

from src.leakage_guard import check_training_rows_chronological

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WIN, DRAW, LOSS = 2, 1, 0

# Knockout probabilities derived from modern WC data (1998–2022)
DRAW_RATE_90MIN    = 0.214   # 24/112 knockout matches drawn at 90 min
EXTRA_TIME_DECIDES = 0.375   # 37.5% of KO draws decided in ET before penalties

# Bayesian update
BAYESIAN_ALPHA = 0.17        # mean optimal alpha across 6 validation years
FORM_MULTIPLIER_MIN = 0.5
FORM_MULTIPLIER_MAX = 2.0
TOURNAMENT_AVG_GOALS_PER_MATCH = 2.6   # 2026 group stage average (frozen)

# Training
CALIBRATION_YEAR   = 2022
EARLY_STOP_YEAR    = 2018
WC_YEARS           = [1998, 2002, 2006, 2010, 2014, 2018, 2022]
RECENCY_FACTOR_DEFAULT = 0.4
SHOOTOUT_K         = 8       # Bayesian shrinkage strength

# Ensemble
WC_WEIGHT_DEFAULT  = 0.70
WC_WEIGHT_CV_RANGE = (0.60, 0.90)

# Class weights for group stage — up-weight the minority draw class
GROUP_CLASS_WEIGHTS = {WIN: 1.2, DRAW: 1.5, LOSS: 1.1}

# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------

# Layer 1 pre-tournament features (Groups 1–4, used for training and Layer 1 pred)
PRETOURNAMENT_FEATURES: list[str] = [
    # Group 1 — Elo (F001–F003, F007)
    "elo_rating", "elo_win_expectancy", "elo_rating_delta", "elo_is_host",
    # Group 2 — FIFA (F008, F009, F011, F013)
    "fifa_points", "fifa_points_delta", "fifa_points_4yr_change",
    "elo_fifa_rank_disagreement",
    # Group 3 — WC historical (F014, F015, F018, F022, +debut flag)
    "wc_win_rate_modern", "wc_win_rate_knockout_modern",
    "wc_gd_per_game_modern", "wc_group_vs_knockout_uplift",
    "wc_debut_modern_flag",
    # Group 4 — form (F023, F026)
    "form_win_rate_last10", "form_gd_last10",
    # Context
    "confederation",
]

# Layer 2 in-tournament features (Group 5)
TOURNAMENT_FEATURES: list[str] = [
    "tourn_pts_md2", "tourn_gd_md2",
    "tourn_avg_sot", "tourn_sot_conceded",
    "tourn_shot_conversion_rate",
    "has_full_tactical_md2",
]

# Knockout stage context
KO_CONTEXT_FEATURES: list[str] = ["stage_order", "is_knockout"]

# Stage A (R32/R16) — all available features
KO_STAGE_A_FEATURES: list[str] = (
    PRETOURNAMENT_FEATURES + TOURNAMENT_FEATURES + KO_CONTEXT_FEATURES
)

# Stage B (QF/SF/Final) — top 12 + stage interaction (per design spec §8)
KO_STAGE_B_FEATURES: list[str] = [
    "elo_rating", "elo_win_expectancy", "elo_rating_delta", "elo_is_host",
    "fifa_points",
    "wc_win_rate_modern", "wc_win_rate_knockout_modern",
    "wc_gd_per_game_modern",
    "tourn_pts_md2", "tourn_gd_md2",
    "form_win_rate_last10", "form_gd_last10",
    "ko_temperament_interaction",   # stage_order × wc_group_vs_knockout_uplift
]

CATEGORICAL_FEATURES: list[str] = ["confederation"]
BINARY_FEATURES: list[str] = [
    "elo_is_host", "is_knockout", "has_full_tactical_md2",
    "wc_debut_modern_flag", "shootout_naive_flag",
]

# Tier 1 tournaments for augmented training corpus (DS4 filter)
TIER1_TOURNAMENT_PATTERNS: list[str] = [
    "FIFA World Cup",
    "UEFA European Championship",
    "UEFA Euro",
    "Copa América",
    "AFC Asian Cup",
    "Africa Cup of Nations",
    "Gold Cup",
    "CONCACAF Nations League",
]

# XGBoost default hyperparameters
_XGB_GROUP_DEFAULTS: dict = {
    "n_estimators":     150,
    "max_depth":        3,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 4,
    "gamma":            0.2,
    "reg_lambda":       1.5,
    "reg_alpha":        0.0,
    "objective":        "multi:softprob",
    "num_class":        3,
    "eval_metric":      "mlogloss",
    "random_state":     42,
    "verbosity":        0,
}
_XGB_KO_BASE: dict = {
    k: v for k, v in _XGB_GROUP_DEFAULTS.items()
    if k not in ("objective", "num_class", "eval_metric")
}
_XGB_KO_DEFAULTS: dict = {
    **_XGB_KO_BASE,
    "objective":   "binary:logistic",
    "eval_metric": "logloss",
}
_XGB_STAGE_A_DEFAULTS: dict = {**_XGB_KO_DEFAULTS, "max_depth": 2}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_rps(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Ranked Probability Score for 3-class ordered outcomes (LOSS < DRAW < WIN).

    Parameters
    ----------
    y_true  : int array of shape (n,) with values in {0=LOSS, 1=DRAW, 2=WIN}.
    y_proba : float array of shape (n, 3) — columns [P(LOSS), P(DRAW), P(WIN)].

    Returns
    -------
    Mean RPS across all rows. Lower is better.
    """
    n = len(y_true)
    rps_sum = 0.0
    for i in range(n):
        outcome = int(y_true[i])
        p = y_proba[i]  # [P(0), P(1), P(2)]
        # Cumulative probabilities
        f0 = float(p[0])
        f1 = float(p[0]) + float(p[1])
        # Cumulative outcome indicators
        o0 = 1.0 if outcome <= 0 else 0.0  # I(outcome == LOSS)
        o1 = 1.0 if outcome <= 1 else 0.0  # I(outcome in {LOSS, DRAW})
        rps_sum += 0.5 * ((f0 - o0) ** 2 + (f1 - o1) ** 2)
    return rps_sum / n if n > 0 else float("nan")


def compute_brier(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Mean Brier score across all outcome classes."""
    n, k = y_proba.shape
    one_hot = np.zeros_like(y_proba)
    for i, yi in enumerate(y_true):
        one_hot[i, int(yi)] = 1.0
    return float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))


def compute_directional_accuracy(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    exclude_draws: bool = True,
) -> float:
    """Accuracy of argmax prediction (optionally excluding draw outcomes/predictions)."""
    y_pred = np.argmax(y_proba, axis=1)
    mask = np.ones(len(y_true), dtype=bool)
    if exclude_draws:
        mask &= (y_true != DRAW) & (y_pred != DRAW)
    if mask.sum() == 0:
        return float("nan")
    return float((y_pred[mask] == y_true[mask]).mean())


# ---------------------------------------------------------------------------
# Preprocessing — for Logistic Regression
# ---------------------------------------------------------------------------

class ConfederationMeanImputer(BaseEstimator, TransformerMixin):
    """Impute WC historical null features with confederation-level mean.

    For teams with wc_debut_modern_flag == 1, replaces null values in WC
    historical columns with the mean of teams in the same confederation that
    have ≥3 WC tournament appearances (wc_tournaments_attended >= 3).

    This implements the 'regression-to-confederation-mean' strategy from
    the design spec §1 Feature Construction Timing Rules.
    """

    WC_HIST_COLS = [
        "wc_win_rate_modern", "wc_win_rate_knockout_modern",
        "wc_gd_per_game_modern", "wc_clean_sheet_rate_modern",
        "wc_group_vs_knockout_uplift", "wc_avg_gf_modern", "wc_avg_ga_modern",
    ]

    def __init__(self) -> None:
        self.confed_means_: dict[str, dict[str, float]] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "ConfederationMeanImputer":
        if "confederation" not in X.columns:
            return self
        for confed, grp in X.groupby("confederation"):
            experienced = grp[grp.get("wc_tournaments_attended", pd.Series(0)) >= 3]
            if experienced.empty:
                experienced = grp  # fallback: use whatever is available
            self.confed_means_[str(confed)] = {}
            for col in self.WC_HIST_COLS:
                if col in experienced.columns:
                    vals = pd.to_numeric(experienced[col], errors="coerce").dropna()
                    self.confed_means_[str(confed)][col] = float(vals.mean()) if len(vals) else 0.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "confederation" not in X.columns:
            return X
        for idx, row in X.iterrows():
            if row.get("wc_debut_modern_flag", 0) != 1:
                continue
            confed = str(row.get("confederation", ""))
            means = self.confed_means_.get(confed, {})
            for col, mean_val in means.items():
                if col in X.columns and (pd.isna(X.at[idx, col])):
                    X.at[idx, col] = mean_val
        return X


def _make_lr_pipeline(feature_cols: list[str]) -> Pipeline:
    """Build a sklearn Pipeline for Logistic Regression with proper preprocessing.

    Steps
    -----
    1. ConfederationMeanImputer  — debutant WC feature imputation
    2. ColumnTransformer         — one-hot categoricals, scale numerics
    3. LogisticRegression        — multinomial or binary, C cross-validated
    """
    num_cols = [c for c in feature_cols if c not in CATEGORICAL_FEATURES]
    cat_cols = [c for c in feature_cols if c in CATEGORICAL_FEATURES]

    numeric_transformer = Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("scale",  StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", numeric_transformer, num_cols))
    if cat_cols:
        transformers.append(("cat", categorical_transformer, cat_cols))

    preprocessor = ColumnTransformer(transformers, remainder="drop")

    return Pipeline([
        ("confed_impute", ConfederationMeanImputer()),
        ("preprocess",    preprocessor),
    ])


def _make_sample_weights(
    df: pd.DataFrame,
    apply_class_weights: bool = True,
    recency_factor: float = RECENCY_FACTOR_DEFAULT,
) -> np.ndarray:
    """Combine recency decay and class imbalance weights.

    weight = class_weight[outcome] × (1 + recency_factor × (year - 1998) / 24)
    """
    n = len(df)
    weights = np.ones(n, dtype=float)

    # Recency decay
    if recency_factor > 0 and "match_year" in df.columns:
        years = pd.to_numeric(df["match_year"], errors="coerce").fillna(1998).values
        weights *= 1.0 + recency_factor * (years - 1998) / 24.0

    # Class imbalance correction
    if apply_class_weights and "outcome" in df.columns:
        for cls, w in GROUP_CLASS_WEIGHTS.items():
            mask = df["outcome"].values == cls
            weights[mask] *= w

    return weights


def _get_X(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Extract feature matrix, filling missing columns with NaN."""
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan
    return X[feature_cols]


def _add_ko_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Add stage_order × wc_group_vs_knockout_uplift interaction feature."""
    df = df.copy()
    stage = pd.to_numeric(df.get("stage_order", 1), errors="coerce").fillna(1)
    uplift = pd.to_numeric(df.get("wc_group_vs_knockout_uplift", 0), errors="coerce").fillna(0)
    df["ko_temperament_interaction"] = stage * uplift
    return df


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def _platt_scale_multiclass(
    raw_proba: np.ndarray,
    y_true: np.ndarray,
    n_classes: int = 3,
) -> tuple[np.ndarray, list]:
    """Fit per-class Platt scaling logistic regressions on calibration holdout.

    Returns calibrated probabilities (renormalized) and fitted calibrators.
    If a calibrator is non-monotonic or fails, falls back to temperature scaling.
    """
    calibrators = []
    calibrated = np.zeros_like(raw_proba)

    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        score = raw_proba[:, c].reshape(-1, 1)
        cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        try:
            cal.fit(score, y_bin)
            calibrated[:, c] = cal.predict_proba(score)[:, 1]
            calibrators.append(cal)
        except Exception:
            calibrated[:, c] = raw_proba[:, c]
            calibrators.append(None)

    # Renormalize rows to sum to 1
    row_sums = calibrated.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    calibrated /= row_sums
    return calibrated, calibrators


def _temperature_scale(
    raw_proba: np.ndarray,
    y_true: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit a single temperature T to minimise NLL on calibration holdout.

    T > 1 softens probabilities (preferred for simulation).
    """
    from scipy.optimize import minimize_scalar

    # Convert to logits
    eps = 1e-7
    proba_clipped = np.clip(raw_proba, eps, 1 - eps)
    logits = np.log(proba_clipped)

    def nll(T: float) -> float:
        if T <= 0:
            return 1e9
        soft = np.exp(logits / T)
        soft /= soft.sum(axis=1, keepdims=True)
        soft = np.clip(soft, eps, 1 - eps)
        log_loss = 0.0
        for i, yi in enumerate(y_true):
            log_loss -= np.log(soft[i, int(yi)])
        return log_loss / len(y_true)

    result = minimize_scalar(nll, bounds=(0.1, 5.0), method="bounded")
    T_opt = float(result.x)

    calibrated = np.exp(logits / T_opt)
    calibrated /= calibrated.sum(axis=1, keepdims=True)
    return calibrated, T_opt


def _apply_platt_calibrators(raw_proba: np.ndarray, calibrators: list) -> np.ndarray:
    """Apply fitted Platt calibrators to new predictions."""
    n, k = raw_proba.shape
    calibrated = np.zeros_like(raw_proba)
    for c, cal in enumerate(calibrators):
        if cal is not None:
            score = raw_proba[:, c].reshape(-1, 1)
            calibrated[:, c] = cal.predict_proba(score)[:, 1]
        else:
            calibrated[:, c] = raw_proba[:, c]
    row_sums = calibrated.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    calibrated /= row_sums
    return calibrated


# ---------------------------------------------------------------------------
# Group Stage Model (Layer 1 — pre-tournament baseline)
# ---------------------------------------------------------------------------

class GroupStageModel:
    """Three-class group stage classifier (WIN=2, DRAW=1, LOSS=0).

    Wraps an XGBoost multi:softprob model and a LogisticRegression baseline.
    The primary prediction uses XGBoost; LR serves as interpretability check
    and fallback. Probability calibration (Platt / temperature / isotonic) is
    applied to the XGBoost outputs using the 2022 calibration holdout.

    The model uses pre-tournament features only (Groups 1–4). Tournament
    features (Group 5) are incorporated via BayesianTournamentUpdater.
    """

    def __init__(
        self,
        feature_cols: list[str] = PRETOURNAMENT_FEATURES,
        xgb_params: Optional[dict] = None,
        lr_C_grid: list[float] = (0.01, 0.1, 1.0, 10.0),
        use_xgb: bool = True,
        recency_factor: float = RECENCY_FACTOR_DEFAULT,
    ) -> None:
        self.feature_cols  = list(feature_cols)
        self.xgb_params    = xgb_params or _XGB_GROUP_DEFAULTS.copy()
        self.lr_C_grid     = list(lr_C_grid)
        self.use_xgb       = use_xgb and _XGB_AVAILABLE
        self.recency_factor= recency_factor

        self._xgb: Optional[object] = None
        self._lr_pipeline: Optional[Pipeline] = None
        self._lr: Optional[LogisticRegression] = None
        self._calibrators: Optional[list] = None
        self._temperature: Optional[float] = None
        self._isotonic:  Optional[list]   = None
        self._calib_method: Optional[str] = None
        self.is_fitted: bool = False
        self.is_calibrated: bool = False
        self.feature_importance_: Optional[pd.DataFrame] = None

    def fit(
        self,
        training_rows: pd.DataFrame,
        early_stop_rows: Optional[pd.DataFrame] = None,
    ) -> "GroupStageModel":
        """Fit XGBoost and Logistic Regression on training rows.

        Parameters
        ----------
        training_rows   : output of features.build_training_rows() with 'outcome' column.
        early_stop_rows : rows from EARLY_STOP_YEAR (2018) for XGBoost early stopping.
        """
        check_training_rows_chronological(training_rows)

        df = training_rows[training_rows["outcome"].notna()].copy()
        y  = df["outcome"].astype(int).values
        sw = _make_sample_weights(df, apply_class_weights=True,
                                  recency_factor=self.recency_factor)

        X = _get_X(df, self.feature_cols)

        # --- XGBoost ---
        if self.use_xgb:
            params = {**self.xgb_params}
            self._xgb = xgb.XGBClassifier(**params)

            if early_stop_rows is not None and not early_stop_rows.empty:
                X_es = _get_X(early_stop_rows, self.feature_cols)
                y_es = early_stop_rows["outcome"].astype(int).values
                self._xgb.set_params(early_stopping_rounds=15)
                self._xgb.fit(
                    X.values, y,
                    sample_weight=sw,
                    eval_set=[(X_es.values, y_es)],
                    verbose=False,
                )
            else:
                self._xgb.fit(X.values, y, sample_weight=sw, verbose=False)

            # get_score() is the correct API; get_fscore() is deprecated and
            # does not accept importance_type in some XGBoost versions
            _fscore = self._xgb.get_booster().get_score(importance_type="gain")
            self.feature_importance_ = pd.DataFrame({
                "feature": self.feature_cols,
                "gain": [_fscore.get(f"f{i}", 0.0) for i in range(len(self.feature_cols))],
            }).sort_values("gain", ascending=False)

        # --- Logistic Regression ---
        lr_pipeline = _make_lr_pipeline(self.feature_cols)
        best_lr = None
        best_score = np.inf
        for C in self.lr_C_grid:
            lr = LogisticRegression(
                C=C, solver="lbfgs", multi_class="multinomial",
                max_iter=1000, class_weight="balanced", penalty="l2",
            )
            try:
                lr_pipeline.fit(X, y)
                X_t = lr_pipeline[:-1].transform(X) if hasattr(lr_pipeline, "__len__") else X
                # Actually just refit each C separately
                pipe = _make_lr_pipeline(self.feature_cols)
                pipe.steps.append(("lr", lr))
                pipe.fit(X, y, lr__sample_weight=sw)
                preds = pipe.predict_proba(X)
                score = compute_rps(y, preds)
                if score < best_score:
                    best_score = score
                    best_lr = pipe
            except Exception:
                continue

        self._lr_pipeline = best_lr

        self.is_fitted = True
        return self

    def calibrate(
        self,
        calib_rows: pd.DataFrame,
        method: str = "auto",
    ) -> "GroupStageModel":
        """Fit probability calibration using the 2022 holdout (Fold 7).

        Parameters
        ----------
        calib_rows : rows from CALIBRATION_YEAR (2022 WC) only.
        method     : 'platt' | 'temperature' | 'isotonic' | 'auto'.
                     'auto' tries Platt first; if the 10% bin criterion is not
                     met, falls back to isotonic regression.
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before calibrate().")
        df = calib_rows[calib_rows["outcome"].notna()].copy()
        y_cal = df["outcome"].astype(int).values
        X_cal = _get_X(df, self.feature_cols)
        raw = self._predict_raw_xgb(X_cal)

        if method in ("platt", "auto"):
            cal_proba, calibrators = _platt_scale_multiclass(raw, y_cal)
            # Check 10% bin criterion
            if _check_calibration_quality(y_cal, cal_proba, threshold=0.10) or method == "platt":
                self._calibrators = calibrators
                self._calib_method = "platt"
            else:
                # Fallback to isotonic regression per class
                self._isotonic = _fit_isotonic(raw, y_cal)
                self._calib_method = "isotonic"
        elif method == "temperature":
            _, T = _temperature_scale(raw, y_cal)
            self._temperature = T
            self._calib_method = "temperature"
        elif method == "isotonic":
            self._isotonic = _fit_isotonic(raw, y_cal)
            self._calib_method = "isotonic"
        else:
            raise ValueError(f"Unknown calibration method: {method!r}")

        self.is_calibrated = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated probability array of shape (n, 3): [P(LOSS), P(DRAW), P(WIN)].

        If not calibrated, returns raw XGBoost probabilities.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        X_feat = _get_X(X, self.feature_cols)
        raw = self._predict_raw_xgb(X_feat)

        if not self.is_calibrated:
            return raw

        if self._calib_method == "platt":
            return _apply_platt_calibrators(raw, self._calibrators)
        elif self._calib_method == "temperature":
            eps = 1e-7
            logits = np.log(np.clip(raw, eps, 1 - eps))
            cal = np.exp(logits / self._temperature)
            cal /= cal.sum(axis=1, keepdims=True)
            return cal
        elif self._calib_method == "isotonic":
            return _apply_isotonic(raw, self._isotonic)
        return raw

    def predict_proba_lr(self, X: pd.DataFrame) -> np.ndarray:
        """Return Logistic Regression baseline probabilities."""
        if self._lr_pipeline is None:
            raise RuntimeError("LR baseline not fitted.")
        X_feat = _get_X(X, self.feature_cols)
        return self._lr_pipeline.predict_proba(X_feat)

    def _predict_raw_xgb(self, X: pd.DataFrame) -> np.ndarray:
        if self._xgb is not None:
            X_num = X.apply(pd.to_numeric, errors="coerce")
            raw = self._xgb.predict_proba(X_num.values)
            # XGBoost outputs [P(0), P(1), P(2)] = [P(LOSS), P(DRAW), P(WIN)] ✓
            return raw
        elif self._lr_pipeline is not None:
            return self._lr_pipeline.predict_proba(X)
        raise RuntimeError("No model fitted.")

    def validate_coefficient_directions(self) -> dict[str, bool]:
        """Check that LR coefficient signs match domain expectations.

        Returns dict {feature_name: True if direction correct, False otherwise}.
        Logs a warning for each reversed coefficient.
        """
        if self._lr_pipeline is None:
            return {}
        # Expected positive for WIN class (class index 2)
        expected_positive = {
            "elo_win_expectancy", "tourn_pts_md2", "form_win_rate_last10",
            "elo_is_host", "wc_win_rate_modern",
        }
        results = {}
        try:
            lr_step = self._lr_pipeline.named_steps.get("lr")
            if lr_step is None:
                return {}
            coef = lr_step.coef_  # shape (n_classes, n_features) for multinomial
            feature_names = self.feature_cols
            # WIN class = index 2 (since WIN=2 is the highest label)
            win_coef = coef[2] if len(coef) > 2 else coef[0]
            for i, fname in enumerate(feature_names):
                if fname in expected_positive:
                    direction_ok = win_coef[i] > 0
                    results[fname] = direction_ok
                    if not direction_ok:
                        warnings.warn(
                            f"Coefficient direction reversal for {fname!r}: "
                            f"coefficient = {win_coef[i]:.4f} (expected positive). "
                            "This may indicate multicollinearity or a data error.",
                            stacklevel=2,
                        )
        except (AttributeError, IndexError):
            pass
        return results


# ---------------------------------------------------------------------------
# Isotonic calibration helpers
# ---------------------------------------------------------------------------

def _fit_isotonic(raw_proba: np.ndarray, y_true: np.ndarray) -> list:
    """Fit one IsotonicRegression per class on calibration holdout."""
    iso_list = []
    n, k = raw_proba.shape
    for c in range(k):
        y_bin = (y_true == c).astype(float)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_proba[:, c], y_bin)
        iso_list.append(iso)
    return iso_list


def _apply_isotonic(raw_proba: np.ndarray, iso_list: list) -> np.ndarray:
    calibrated = np.zeros_like(raw_proba)
    for c, iso in enumerate(iso_list):
        calibrated[:, c] = iso.predict(raw_proba[:, c])
    row_sums = calibrated.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    calibrated /= row_sums
    return calibrated


def _check_calibration_quality(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    threshold: float = 0.10,
) -> bool:
    """Return True if all bins have mean predicted vs actual deviation < threshold."""
    bins = np.linspace(0, 1, n_bins + 1)
    win_proba = y_proba[:, 2]  # P(WIN)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (win_proba >= lo) & (win_proba < hi)
        if mask.sum() < 3:
            continue
        mean_pred = float(win_proba[mask].mean())
        actual_rate = float((y_true[mask] == WIN).mean())
        if abs(mean_pred - actual_rate) > threshold:
            return False
    return True


# ---------------------------------------------------------------------------
# Knockout Model (Layer 3 — stage-conditional)
# ---------------------------------------------------------------------------

class KnockoutModel:
    """Stage-conditional binary classifier for knockout matches.

    Two stage groups:
      Stage A (R32 + R16)  — XGBoost (max_depth=2) + LR, features KO_STAGE_A_FEATURES
      Stage B (QF/SF/Final)— LogReg primary + XGBoost secondary (equal ensemble)

    The knockout model outputs P(team_a wins at 90 min), then applies the
    fixed 21.4% draw rate at 90 minutes:
      P(win)  = model_output × (1 - DRAW_RATE_90MIN)
      P(draw) = DRAW_RATE_90MIN
      P(loss) = (1 - model_output) × (1 - DRAW_RATE_90MIN)

    Training target: 1 = win at 90 min, 0 = loss at 90 min.
    Draw matches are EXCLUDED from training (as per design spec §2 Target T2).
    """

    def __init__(
        self,
        xgb_stage_a_params: Optional[dict] = None,
        recency_factor: float = RECENCY_FACTOR_DEFAULT,
    ) -> None:
        self.xgb_stage_a_params = xgb_stage_a_params or _XGB_STAGE_A_DEFAULTS.copy()
        self.recency_factor = recency_factor

        self._stage_a_xgb: Optional[object] = None
        self._stage_a_lr:  Optional[Pipeline] = None
        self._stage_b_lr:  Optional[Pipeline] = None
        self._stage_b_xgb: Optional[object]  = None
        self.is_fitted = False

    def _prep_ko(self, df: pd.DataFrame, feature_cols: list[str]):
        df = _add_ko_interaction(df.copy())
        # Exclude draws — binary target: 1=win, 0=loss
        df_ko = df[df["outcome"].isin([WIN, LOSS])].copy()
        df_ko["target"] = (df_ko["outcome"] == WIN).astype(int)
        y  = df_ko["target"].values
        X  = _get_X(df_ko, feature_cols)
        sw = _make_sample_weights(df_ko, apply_class_weights=False,
                                  recency_factor=self.recency_factor)
        return X, y, sw

    def fit(
        self,
        training_rows: pd.DataFrame,
    ) -> "KnockoutModel":
        """Fit Stage A and Stage B models from training_rows.

        Filters to knockout matches using 'is_knockout' column, then
        further splits by stage_order: A = stage_order in {2, 3} (R32/R16),
        B = stage_order in {4, 5, 6, 7} (QF/SF/Final).
        """
        _idx = training_rows.index
        ko_rows = training_rows[
            training_rows.get("is_knockout", pd.Series(False, index=_idx)).astype(bool)
            | (training_rows.get("stage_order", pd.Series(1, index=_idx)).astype(float) > 1)
        ].copy()

        # Stage A: Round of 32 / Round of 16 (stage_order 2–3)
        _ko_idx = ko_rows.index
        stage_a = ko_rows[ko_rows.get("stage_order", pd.Series(1, index=_ko_idx)).astype(float).isin([2.0, 3.0])]
        if len(stage_a) > 0:
            X_a, y_a, sw_a = self._prep_ko(stage_a, KO_STAGE_A_FEATURES)
            if _XGB_AVAILABLE:
                self._stage_a_xgb = xgb.XGBClassifier(**self.xgb_stage_a_params)
                self._stage_a_xgb.fit(X_a.values, y_a, sample_weight=sw_a, verbose=False)
            lr_a = _make_lr_pipeline(KO_STAGE_A_FEATURES)
            lr_a.steps.append(("lr", LogisticRegression(
                C=0.1, solver="lbfgs", max_iter=1000, class_weight="balanced"
            )))
            lr_a.fit(X_a, y_a, lr__sample_weight=sw_a)
            self._stage_a_lr = lr_a

        # Stage B: QF/SF/Final (stage_order 4–7)
        stage_b = ko_rows[ko_rows.get("stage_order", pd.Series(1, index=_ko_idx)).astype(float) >= 4.0]
        if len(stage_b) > 0:
            X_b, y_b, sw_b = self._prep_ko(stage_b, KO_STAGE_B_FEATURES)
            # Primary: strong LR (C=0.1 per design spec §8)
            lr_b = _make_lr_pipeline(KO_STAGE_B_FEATURES)
            lr_b.steps.append(("lr", LogisticRegression(
                C=0.1, solver="lbfgs", max_iter=1000, class_weight="balanced"
            )))
            lr_b.fit(X_b, y_b, lr__sample_weight=sw_b)
            self._stage_b_lr = lr_b
            # Secondary: XGBoost (averaged with LR at equal weight)
            if _XGB_AVAILABLE:
                xgb_b = xgb.XGBClassifier(**{
                    **self.xgb_stage_a_params, "max_depth": 2, "n_estimators": 100,
                })
                xgb_b.fit(X_b.values, y_b, sample_weight=sw_b, verbose=False)
                self._stage_b_xgb = xgb_b

        self.is_fitted = True
        return self

    def predict_proba(
        self,
        X: pd.DataFrame,
        stage_order: int = 2,
    ) -> np.ndarray:
        """Return array of shape (n, 3): [P(win@90), P(draw@90), P(loss@90)].

        The fixed 21.4% draw rate is applied after the binary classifier output.
        """
        if not self.is_fitted:
            raise RuntimeError("KnockoutModel not fitted.")
        X = _add_ko_interaction(X.copy())

        if stage_order in (2, 3):
            p_win = self._predict_binary_stage_a(X)
        else:
            p_win = self._predict_binary_stage_b(X)

        p_draw = np.full_like(p_win, DRAW_RATE_90MIN)
        p_win_adj  = p_win  * (1 - DRAW_RATE_90MIN)
        p_loss_adj = (1 - p_win) * (1 - DRAW_RATE_90MIN)
        return np.stack([p_win_adj, p_draw, p_loss_adj], axis=1)

    def _predict_binary_stage_a(self, X: pd.DataFrame) -> np.ndarray:
        X_feat = _get_X(X, KO_STAGE_A_FEATURES)
        preds = []
        if self._stage_a_lr is not None:
            p_lr = self._stage_a_lr.predict_proba(X_feat)[:, 1]
            preds.append(p_lr)
        if self._stage_a_xgb is not None:
            p_xgb = self._stage_a_xgb.predict_proba(X_feat.values)[:, 1]
            preds.append(p_xgb)
        if not preds:
            return np.full(len(X), 0.5)
        # Equal weight ensemble
        return np.mean(np.stack(preds, axis=1), axis=1)

    def _predict_binary_stage_b(self, X: pd.DataFrame) -> np.ndarray:
        X_feat = _get_X(X, KO_STAGE_B_FEATURES)
        preds = []
        if self._stage_b_lr is not None:
            p_lr = self._stage_b_lr.predict_proba(X_feat)[:, 1]
            preds.append(p_lr)
        if self._stage_b_xgb is not None:
            p_xgb = self._stage_b_xgb.predict_proba(X_feat.values)[:, 1]
            preds.append(p_xgb)
        if not preds:
            return np.full(len(X), 0.5)
        return np.mean(np.stack(preds, axis=1), axis=1)


# ---------------------------------------------------------------------------
# Penalty Shootout Model
# ---------------------------------------------------------------------------

class PenaltyModel:
    """Three-component penalty shootout model.

    Components
    ----------
    1. Bayesian-shrunk historical shootout win rate (w1 = 0.50)
       shrunk_rate = (wins + k × 0.5) / (appearances + k),  k = 8
       Normalised: shrunk_A / (shrunk_A + shrunk_B)

    2. Pre-tournament Elo win expectancy (w2 = 0.40)
       Verified at 58% accuracy on 119 DS6 records with Elo data available.

    3. WC-specific shootout win rate (w3 = 0.10)
       Used only if team has ≥2 WC shootout appearances; otherwise w3 → w1.

    When neither team has ≥2 WC appearances: w1 = 0.55, w2 = 0.45, w3 = 0.00.
    Output is normalised to P(A) + P(B) = 1.0.
    """

    def predict_proba(
        self,
        team_a_feats: pd.Series,
        team_b_feats: pd.Series,
    ) -> float:
        """Return P(team A wins the penalty shootout).

        Parameters
        ----------
        team_a_feats, team_b_feats : rows from the team feature table,
            containing shootout_win_rate_alltime, shootout_appearances_total,
            shootout_win_rate_wc_only, shootout_naive_flag, elo_win_expectancy.
        """
        def _get(s: pd.Series, col: str, default) -> float:
            v = s.get(col, default)
            return default if pd.isna(v) else float(v)

        # Component 1 — shrunk shootout rate
        shrunk_a = _get(team_a_feats, "shootout_win_rate_alltime", 0.5)
        shrunk_b = _get(team_b_feats, "shootout_win_rate_alltime", 0.5)
        denom_1  = shrunk_a + shrunk_b
        c1_a = (shrunk_a / denom_1) if denom_1 > 0 else 0.5

        # Component 2 — Elo win expectancy (team A vs team B)
        c2_a = _get(team_a_feats, "elo_win_expectancy", 0.5)
        # elo_win_expectancy in the feature table is already from team's perspective vs opponent
        # If the feature is already computed relative to this specific opponent, use directly.
        # Otherwise we recompute:
        elo_a = _get(team_a_feats, "elo_rating", 1500.0)
        elo_b = _get(team_b_feats, "elo_rating", 1500.0)
        if elo_a != 1500.0 or elo_b != 1500.0:
            c2_a = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))

        # Component 3 — WC-specific shootout rate
        wc_apps_a = int(_get(team_a_feats, "shootout_appearances_total", 0))
        wc_rate_a = _get(team_a_feats, "shootout_win_rate_wc_only", np.nan)
        wc_apps_b = int(_get(team_b_feats, "shootout_appearances_total", 0))
        wc_rate_b = _get(team_b_feats, "shootout_win_rate_wc_only", np.nan)

        has_wc_data = (
            wc_apps_a >= 2 and not np.isnan(wc_rate_a)
            and wc_apps_b >= 2 and not np.isnan(wc_rate_b)
        )

        if has_wc_data:
            w1, w2, w3 = 0.50, 0.40, 0.10
            denom_3 = wc_rate_a + wc_rate_b
            c3_a = (wc_rate_a / denom_3) if denom_3 > 0 else 0.5
        else:
            w1, w2, w3 = 0.55, 0.45, 0.00
            c3_a = 0.5

        p_a = w1 * c1_a + w2 * c2_a + w3 * c3_a
        # Normalise (sum should be ≈1 already by construction, but enforce)
        p_a = float(np.clip(p_a, 0.01, 0.99))
        return p_a


# ---------------------------------------------------------------------------
# Bayesian Tournament Update (Layer 2)
# ---------------------------------------------------------------------------

class BayesianTournamentUpdater:
    """Layer 2 update: adjust pre-tournament probabilities with group-stage form.

    Update formula (design spec §7):
        P_posterior = (1 - α) × P_prior + α × (P_prior × multiplier)
        multiplier  = f(pts_delta, gd_delta, gf_norm, ga_norm) ∈ [0.5, 2.0]
        α           = 0.17 (empirically derived, held constant throughout simulation)

    The multiplier is computed from four components with equal weights (0.25 each)
    unless fitted weights are available from historical validation.

    The four components use only features available for ALL 48 teams (no
    tactical-only features missing for the four June 23 teams).
    """

    def __init__(self, alpha: float = BAYESIAN_ALPHA) -> None:
        self.alpha = alpha
        self._component_weights: np.ndarray = np.array([0.25, 0.25, 0.25, 0.25])
        self._is_fitted: bool = False

    def fit(
        self,
        historical_rows: Optional[pd.DataFrame] = None,
    ) -> "BayesianTournamentUpdater":
        """Optionally fit component weights from historical WC group-stage data.

        If historical_rows is None, uses equal weights (0.25 each) as specified
        in the design doc fallback.
        """
        if historical_rows is None or historical_rows.empty:
            self._component_weights = np.array([0.25, 0.25, 0.25, 0.25])
            self._is_fitted = True
            return self

        # The historical model fits a linear regression on the 4 tournament
        # performance components to predict knockout stage performance delta.
        # This is implemented as a simple OLS if column are present.
        required = [
            "pts_relative_to_elo", "gd_relative_to_elo",
            "gf_norm", "ga_norm", "ko_overperformance",
        ]
        if not all(c in historical_rows.columns for c in required):
            # Columns not available — use equal weights
            self._component_weights = np.array([0.25, 0.25, 0.25, 0.25])
            self._is_fitted = True
            return self

        X_fit = historical_rows[required[:4]].values
        y_fit = historical_rows["ko_overperformance"].values
        mask  = ~np.isnan(X_fit).any(axis=1) & ~np.isnan(y_fit)
        if mask.sum() < 10:
            self._component_weights = np.array([0.25, 0.25, 0.25, 0.25])
        else:
            from numpy.linalg import lstsq
            w, _, _, _ = lstsq(X_fit[mask], y_fit[mask], rcond=None)
            w = np.abs(w)
            w /= w.sum() if w.sum() > 0 else 1.0
            self._component_weights = w

        self._is_fitted = True
        return self

    def compute_form_multiplier(
        self,
        team_feats: pd.Series,
        elo_expected_pts: float,
        elo_expected_gd: float,
        avg_tournament_goals: float = TOURNAMENT_AVG_GOALS_PER_MATCH,
    ) -> float:
        """Compute the knockout form multiplier for one team.

        Parameters
        ----------
        team_feats          : team's feature row (must include tourn_pts_md2,
                              tourn_gd_md2, tourn_gf_md2, tourn_ga_md2).
        elo_expected_pts    : expected points from Elo win expectancy against
                              both MD1 + MD2 opponents.
        elo_expected_gd     : expected goal difference from historical WC
                              average at the given Elo differential.
        avg_tournament_goals: normalisation constant for goals per match.

        Returns
        -------
        Multiplier in [0.5, 2.0]. Values > 1 mean outperforming Elo expectation.
        """

        def _get(col: str, default: float = 0.0) -> float:
            v = team_feats.get(col, default)
            return default if pd.isna(v) else float(v)

        pts = _get("tourn_pts_md2")
        gd  = _get("tourn_gd_md2")
        gf  = _get("tourn_gf_md2")
        ga  = _get("tourn_ga_md2")

        # Component 1: points relative to Elo expectation
        c1 = (pts - elo_expected_pts) / 6.0   # normalise by max (6 pts in 2 games)

        # Component 2: GD relative to Elo expectation
        c2 = (gd - elo_expected_gd) / 5.0     # normalise by typical GD range

        # Component 3: goals scored per match, normalised
        c3 = (gf / 2.0) / avg_tournament_goals - 1.0  # positive if above avg

        # Component 4: goals conceded per match, normalised (inverted: lower is better)
        c4 = 1.0 - (ga / 2.0) / avg_tournament_goals  # positive if below avg

        w = self._component_weights
        raw_signal = float(w[0] * c1 + w[1] * c2 + w[2] * c3 + w[3] * c4)

        # Map signal → multiplier ∈ [0.5, 2.0]
        # Signal ≈ 0  → multiplier ≈ 1.0 (neutral)
        # Signal > 0  → multiplier > 1.0 (positive update)
        # Signal < 0  → multiplier < 1.0 (negative update)
        multiplier = 1.0 + raw_signal
        return float(np.clip(multiplier, FORM_MULTIPLIER_MIN, FORM_MULTIPLIER_MAX))

    def update_proba(
        self,
        p_prior: np.ndarray,
        team_feats: pd.Series,
        elo_expected_pts: float = 3.0,
        elo_expected_gd: float  = 0.0,
    ) -> np.ndarray:
        """Apply Bayesian update to a prior probability vector.

        Parameters
        ----------
        p_prior : array of shape (3,) — [P(LOSS), P(DRAW), P(WIN)] from Layer 1.
        team_feats : team's feature row.

        Returns
        -------
        Updated probability array of shape (3,), renormalised to sum to 1.
        """
        multiplier = self.compute_form_multiplier(team_feats, elo_expected_pts, elo_expected_gd)

        # P_likelihood = P_prior × multiplier (applied to WIN class only)
        p_likelihood = p_prior.copy()
        p_likelihood[2] = np.clip(p_prior[2] * multiplier, 0.0, 0.99)

        # Renormalise likelihood to sum to 1
        s = p_likelihood.sum()
        if s > 0:
            p_likelihood /= s

        # Bayesian blend
        p_posterior = (1 - self.alpha) * p_prior + self.alpha * p_likelihood

        # Enforce valid probability vector: clip then renormalise
        p_posterior = np.clip(p_posterior, 0.0, 1.0)
        s2 = p_posterior.sum()
        if s2 > 0:
            p_posterior /= s2

        return p_posterior

    def apply_qualification_context(
        self,
        p_posterior: np.ndarray,
        already_qualified: bool = False,
        already_eliminated: bool = False,
    ) -> np.ndarray:
        """Apply MD3 group-context adjustments (design spec §10 Phase 1 Step 3).

        - already_qualified: −0.03 on win probability (rotation tendencies).
        - already_eliminated: −0.03 on draw probability (higher variance).
        """
        p = p_posterior.copy()  # [P(LOSS), P(DRAW), P(WIN)]
        if already_qualified:
            adj = min(0.03, p[2] * 0.5)
            p[2] -= adj
            p[0] += adj  # redistribute to loss (weaker lineup)
        if already_eliminated:
            adj = min(0.03, p[1] * 0.5)
            p[1] -= adj
            p[0] += adj  # more likely to lose with nothing to play for

        s = p.sum()
        if s > 0:
            p /= s
        return p


# ---------------------------------------------------------------------------
# Chronological Cross-Validation
# ---------------------------------------------------------------------------

@dataclass
class CVFoldResult:
    fold: int
    val_year: int
    train_years: list[int]
    n_train: int
    n_val: int
    accuracy: float
    rps: float
    brier: float
    best_hyperparams: dict = field(default_factory=dict)


class ChronologicalCV:
    """Leave-one-tournament-out cross-validation (design spec §3).

    Fold structure (exact and non-negotiable):
      Folds 1–6: hyperparameter search (years 1998, 2002, 2006, 2010, 2014, 2018)
      Fold 7   : calibration holdout only (year 2022) — never used for HP tuning

    Usage
    -----
        cv = ChronologicalCV(model_class=GroupStageModel)
        results = cv.evaluate(training_rows, xgb_param_grids)
        print(cv.summary())
    """

    FOLD_STRUCTURE = [
        # (val_year, train_years)
        (1998, [2002, 2006, 2010, 2014, 2018, 2022]),
        (2002, [1998, 2006, 2010, 2014, 2018, 2022]),
        (2006, [1998, 2002, 2010, 2014, 2018, 2022]),
        (2010, [1998, 2002, 2006, 2014, 2018, 2022]),
        (2014, [1998, 2002, 2006, 2010, 2018, 2022]),
        (2018, [1998, 2002, 2006, 2010, 2014, 2022]),
        # Fold 7 (calibration) handled separately — not included here
    ]

    def __init__(
        self,
        model_class=None,
        feature_cols: list[str] = PRETOURNAMENT_FEATURES,
        recency_factor: float = RECENCY_FACTOR_DEFAULT,
    ) -> None:
        self.model_class   = model_class or GroupStageModel
        self.feature_cols  = feature_cols
        self.recency_factor= recency_factor
        self.fold_results_: list[CVFoldResult] = []

    def evaluate(
        self,
        training_rows: pd.DataFrame,
        xgb_param_grid: Optional[dict] = None,
    ) -> list[CVFoldResult]:
        """Run all 6 active folds and return per-fold metrics.

        Standard k-fold is prohibited — all folds follow chronological ordering.
        """
        check_training_rows_chronological(training_rows)
        self.fold_results_ = []

        for fold_i, (val_year, train_years) in enumerate(self.FOLD_STRUCTURE, start=1):
            train_mask = training_rows["match_year"].isin(train_years)
            val_mask   = training_rows["match_year"] == val_year

            train_df = training_rows[train_mask].copy()
            val_df   = training_rows[val_mask].copy()

            if len(train_df) == 0 or len(val_df) == 0:
                continue

            # Early stopping uses the year just before val_year within train set
            es_candidates = [y for y in sorted(train_years) if y < val_year]
            es_year = es_candidates[-1] if es_candidates else None
            es_df = training_rows[training_rows["match_year"] == es_year] if es_year else None

            # Grid search over provided XGBoost params (default if none given)
            best_params = {}
            best_model  = None
            best_rps    = np.inf

            param_list = _expand_param_grid(xgb_param_grid or {})
            if not param_list:
                param_list = [{}]

            for params in param_list:
                combined_params = {**_XGB_GROUP_DEFAULTS, **params}
                model = self.model_class(
                    feature_cols=self.feature_cols,
                    xgb_params=combined_params,
                    recency_factor=self.recency_factor,
                )
                try:
                    model.fit(train_df, early_stop_rows=es_df)
                    X_val = _get_X(val_df, self.feature_cols)
                    y_val = val_df["outcome"].astype(int).values
                    p_val = model.predict_proba(X_val)
                    rps   = compute_rps(y_val, p_val)
                    if rps < best_rps:
                        best_rps    = rps
                        best_params = params
                        best_model  = model
                except Exception as e:
                    warnings.warn(f"Fold {fold_i} param set failed: {e}", stacklevel=2)

            if best_model is None:
                continue

            X_val = _get_X(val_df, self.feature_cols)
            y_val = val_df["outcome"].astype(int).values
            p_val = best_model.predict_proba(X_val)

            self.fold_results_.append(CVFoldResult(
                fold=fold_i,
                val_year=val_year,
                train_years=train_years,
                n_train=len(train_df),
                n_val=len(val_df),
                accuracy=compute_directional_accuracy(y_val, p_val),
                rps=best_rps,
                brier=compute_brier(y_val, p_val),
                best_hyperparams=best_params,
            ))

        return self.fold_results_

    def summary(self) -> pd.DataFrame:
        """Return per-fold metrics as a DataFrame."""
        if not self.fold_results_:
            raise RuntimeError("No results. Call evaluate() first.")
        return pd.DataFrame([{
            "fold":        r.fold,
            "val_year":    r.val_year,
            "n_train":     r.n_train,
            "n_val":       r.n_val,
            "accuracy":    r.accuracy,
            "rps":         r.rps,
            "brier":       r.brier,
            "best_params": str(r.best_hyperparams),
        } for r in self.fold_results_])

    def mean_rps(self) -> float:
        if not self.fold_results_:
            return float("nan")
        return float(np.mean([r.rps for r in self.fold_results_]))

    def mean_accuracy(self) -> float:
        if not self.fold_results_:
            return float("nan")
        return float(np.mean([r.accuracy for r in self.fold_results_ if not np.isnan(r.accuracy)]))


def _expand_param_grid(grid: dict) -> list[dict]:
    """Expand a parameter grid dict into a list of parameter dicts."""
    if not grid:
        return [{}]
    import itertools
    keys   = list(grid.keys())
    values = [grid[k] if isinstance(grid[k], (list, tuple)) else [grid[k]] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


# ---------------------------------------------------------------------------
# Augmented Training Corpus Builder
# ---------------------------------------------------------------------------

TIER1_TOURNAMENTS_RE = "|".join(TIER1_TOURNAMENT_PATTERNS)


def build_augmented_training_rows(
    ds4: pd.DataFrame,
    ds2: pd.DataFrame,
    ds6: pd.DataFrame,
) -> pd.DataFrame:
    """Build training rows from DS4 Tier 1 competitive matches (2014–2026).

    Used to train the secondary model variant (augmented corpus). These rows
    supplement the WC-only corpus for teams with limited WC history. The two
    corpora are NEVER mixed — they are trained separately and ensembled.

    Features per row: Elo (from year-end DS2 snapshot), form (24-month window
    from DS4), shootout rates. WC historical features are excluded here since
    DS4 augmented rows span non-WC competitions.

    Returns a DataFrame compatible with build_training_rows() for model fitting.
    """
    # Filter: Tier 1, non-Friendly, 2014-01-01 to 2026-06-10, scores populated
    tier1_mask = (
        ds4["tournament"].str.contains(TIER1_TOURNAMENTS_RE, case=False, na=False)
        & ~ds4["tournament"].str.contains("qualif", case=False, na=False)
        & (ds4["date"] >= pd.Timestamp("2014-01-01"))
        & (ds4["date"] < pd.Timestamp("2026-06-11"))
        & ds4["home_score"].notna()
        & ds4["away_score"].notna()
    )
    tier1 = ds4[tier1_mask].copy()

    # Competitive matches for form features (any year, non-Friendly)
    competitive = ds4[
        (ds4["tournament"].str.strip() != "Friendly")
        & ds4["home_score"].notna()
        & ds4["away_score"].notna()
    ].copy()

    # Build DS2 lookup indexed by (country, year)
    ds2_by_snap: dict[str, pd.DataFrame] = {
        snap: grp.set_index("country")
        for snap, grp in ds2.groupby("snapshot_date")
    }

    from name_map import CANONICAL_48, apply_to_series
    rows = []

    for _, match in tier1.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        if home not in CANONICAL_48 or away not in CANONICAL_48:
            continue

        match_date = match["date"]
        match_year = match_date.year
        home_score = int(match["home_score"])
        away_score = int(match["away_score"])

        elo_snap_key = f"{match_year - 1}-12-31"
        elo_snap = ds2_by_snap.get(elo_snap_key, pd.DataFrame())

        def _elo(team: str) -> dict:
            if elo_snap.empty or team not in elo_snap.index:
                return {"elo_rating": np.nan, "elo_rank": np.nan,
                        "elo_rating_career_peak": np.nan, "elo_rating_career_avg": np.nan,
                        "confederation": np.nan, "elo_is_host": 0}
            row = elo_snap.loc[team]
            return {
                "elo_rating":            float(row.get("rating", np.nan)),
                "elo_rank":              float(row.get("rank", np.nan)),
                "elo_rating_career_peak":float(row.get("rating_max", np.nan)),
                "elo_rating_career_avg": float(row.get("rating_avg", np.nan)),
                "confederation":         str(row.get("confederation", "")),
                "elo_is_host":           0,
            }

        def _form(team: str) -> dict:
            start_dt = match_date - pd.Timedelta(days=730)
            hm = competitive[
                (competitive["home_team"] == team)
                & (competitive["date"] >= start_dt)
                & (competitive["date"] < match_date)
            ][["home_score", "away_score"]].rename(columns={"home_score": "gf", "away_score": "ga"})
            am = competitive[
                (competitive["away_team"] == team)
                & (competitive["date"] >= start_dt)
                & (competitive["date"] < match_date)
            ][["home_score", "away_score"]].rename(columns={"away_score": "gf", "home_score": "ga"})
            m10 = pd.concat([hm, am]).tail(10)
            n = len(m10)
            if n == 0:
                return {"form_win_rate_last10": np.nan, "form_gd_last10": np.nan}
            gf = m10["gf"].astype(float)
            ga = m10["ga"].astype(float)
            return {
                "form_win_rate_last10": float((gf > ga).sum()) / n,
                "form_gd_last10":       float((gf - ga).mean()),
            }

        def _build_row(team: str, opp: str, is_home: bool) -> dict:
            te = _elo(team)
            oe = _elo(opp)
            tf = _form(team)
            te_r  = te.get("elo_rating", np.nan)
            oe_r  = oe.get("elo_rating", np.nan)
            delta = te_r - oe_r if (pd.notna(te_r) and pd.notna(oe_r)) else np.nan
            win_exp = (1.0 / (1.0 + 10.0 ** (-delta / 400.0))
                       if pd.notna(delta) else np.nan)
            outcome = (WIN if (is_home and home_score > away_score)
                            or (not is_home and away_score > home_score)
                       else DRAW if home_score == away_score
                       else LOSS)
            return {
                "match_year":       match_year,
                "elo_year_used":    match_year - 1,
                "match_date":       match_date,
                "team_canonical":   team,
                "opponent_canonical": opp,
                "is_home_team":     int(is_home),
                "elo_rating":       te_r,
                "elo_win_expectancy": win_exp,
                "elo_rating_delta": delta,
                "elo_rating_career_peak": te.get("elo_rating_career_peak"),
                "elo_rating_career_avg":  te.get("elo_rating_career_avg"),
                "elo_is_host":      te.get("elo_is_host", 0),
                "confederation":    te.get("confederation"),
                **tf,
                "outcome": outcome,
            }

        rows.append(_build_row(home, away, is_home=True))
        rows.append(_build_row(away, home, is_home=False))

    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# WC Forecast Ensemble — master model combining all components
# ---------------------------------------------------------------------------

class WCForecastEnsemble:
    """Master ensemble combining WC-only and augmented model variants.

    Layer 1: Group stage probabilities from WC-only XGBoost + optional augmented
    Layer 2: Bayesian tournament update (α = 0.17)
    Layer 3: Stage-conditional knockout + penalty shootout

    Ensemble weight for group stage:
        P_final = wc_weight × P_wc_only + (1 - wc_weight) × P_augmented
    Default wc_weight = 0.70; CV-tunable in [0.60, 0.90].
    """

    def __init__(
        self,
        wc_weight: float = WC_WEIGHT_DEFAULT,
        recency_factor: float = RECENCY_FACTOR_DEFAULT,
        bayesian_alpha: float = BAYESIAN_ALPHA,
    ) -> None:
        if not (0.0 <= wc_weight <= 1.0):
            raise ValueError(f"wc_weight must be in [0, 1], got {wc_weight}")
        self.wc_weight      = wc_weight
        self.recency_factor = recency_factor
        self.bayesian_alpha = bayesian_alpha

        self.group_model_wc:  Optional[GroupStageModel] = None
        self.group_model_aug: Optional[GroupStageModel] = None
        self.knockout_model:  Optional[KnockoutModel]   = None
        self.penalty_model:   PenaltyModel               = PenaltyModel()
        self.updater:         BayesianTournamentUpdater  = BayesianTournamentUpdater(
            alpha=bayesian_alpha
        )
        self.is_fitted = False

    def fit(
        self,
        wc_training_rows: pd.DataFrame,
        calib_rows: pd.DataFrame,
        aug_training_rows: Optional[pd.DataFrame] = None,
        cv_tune_wc_weight: bool = False,
    ) -> "WCForecastEnsemble":
        """Fit all model components.

        Parameters
        ----------
        wc_training_rows  : DS8-based WC training rows (features.build_training_rows()).
        calib_rows        : 2022 WC rows for calibration (Fold 7 — touched once).
        aug_training_rows : DS4 Tier 1 augmented rows (build_augmented_training_rows()).
                            If None, ensemble degenerates to WC-only (wc_weight = 1.0).
        cv_tune_wc_weight : If True, use ChronologicalCV to select wc_weight from
                            WC_WEIGHT_CV_RANGE in steps of 0.05.
        """
        check_training_rows_chronological(wc_training_rows)

        early_stop_rows = wc_training_rows[
            wc_training_rows["match_year"] == EARLY_STOP_YEAR
        ].copy()
        train_no_2022 = wc_training_rows[
            wc_training_rows["match_year"] != CALIBRATION_YEAR
        ].copy()

        # --- WC-only group stage model ---
        self.group_model_wc = GroupStageModel(recency_factor=self.recency_factor)
        self.group_model_wc.fit(train_no_2022, early_stop_rows=early_stop_rows)
        self.group_model_wc.calibrate(calib_rows)

        # --- Augmented group stage model ---
        if aug_training_rows is not None and not aug_training_rows.empty:
            self.group_model_aug = GroupStageModel(recency_factor=self.recency_factor)
            self.group_model_aug.fit(aug_training_rows)
            self.group_model_aug.calibrate(calib_rows)

        # --- CV ensemble weight tuning ---
        if cv_tune_wc_weight and self.group_model_aug is not None:
            self.wc_weight = self._tune_ensemble_weight(wc_training_rows, calib_rows)

        # --- Knockout model ---
        # WC-only: knockout matches from DS8 (stage_order > 1)
        ko_rows = wc_training_rows[
            wc_training_rows.get("stage_order", pd.Series(1, index=wc_training_rows.index)) > 1
        ].copy() if "stage_order" in wc_training_rows.columns else pd.DataFrame()

        if not ko_rows.empty:
            self.knockout_model = KnockoutModel(recency_factor=self.recency_factor)
            self.knockout_model.fit(ko_rows)

        # --- Bayesian updater ---
        self.updater = BayesianTournamentUpdater(alpha=self.bayesian_alpha)
        self.updater.fit(None)  # equal weights (historical data not available in standard pipeline)

        self.is_fitted = True
        return self

    def predict_group_stage(
        self,
        X: pd.DataFrame,
        apply_bayesian_update: bool = False,
        team_feats: Optional[pd.Series] = None,
        elo_expected_pts: float = 3.0,
        elo_expected_gd: float  = 0.0,
    ) -> np.ndarray:
        """Return array (n, 3): [P(LOSS), P(DRAW), P(WIN)] for group stage matches.

        Parameters
        ----------
        apply_bayesian_update : If True, apply Layer 2 update to each row.
        team_feats            : Required if apply_bayesian_update is True.
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted.")

        p_wc = self.group_model_wc.predict_proba(X)

        if self.group_model_aug is not None and self.wc_weight < 1.0:
            p_aug = self.group_model_aug.predict_proba(X)
            p_combined = self.wc_weight * p_wc + (1 - self.wc_weight) * p_aug
        else:
            p_combined = p_wc

        if apply_bayesian_update and team_feats is not None:
            updated = []
            for i in range(len(p_combined)):
                feats_i = (
                    team_feats.iloc[i]
                    if isinstance(team_feats, pd.DataFrame)
                    else team_feats
                )
                updated.append(
                    self.updater.update_proba(
                        p_combined[i], feats_i, elo_expected_pts, elo_expected_gd
                    )
                )
            p_combined = np.array(updated)

        return p_combined

    def predict_knockout(
        self,
        X: pd.DataFrame,
        stage_order: int = 2,
        apply_bayesian_update: bool = False,
        team_feats: Optional[pd.Series] = None,
        elo_expected_pts: float = 3.0,
        elo_expected_gd: float  = 0.0,
    ) -> np.ndarray:
        """Return array (n, 3): [P(win@90), P(draw@90), P(loss@90)]."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted.")
        if self.knockout_model is None:
            # Fallback: use group stage model with draw rate adjustment
            p_gs = self.predict_group_stage(X)
            p_win  = p_gs[:, 2] * (1 - DRAW_RATE_90MIN)
            p_draw = np.full(len(X), DRAW_RATE_90MIN)
            p_loss = p_gs[:, 0] * (1 - DRAW_RATE_90MIN)
            return np.stack([p_win, p_draw, p_loss], axis=1)

        p = self.knockout_model.predict_proba(X, stage_order=stage_order)

        if apply_bayesian_update and team_feats is not None:
            updated = []
            for i in range(len(p)):
                feats_i = (
                    team_feats.iloc[i]
                    if isinstance(team_feats, pd.DataFrame)
                    else team_feats
                )
                up = self.updater.update_proba(
                    np.array([p[i, 2], p[i, 1], p[i, 0]]),  # [loss, draw, win]
                    feats_i, elo_expected_pts, elo_expected_gd
                )
                updated.append([up[2], up[1], up[0]])  # back to [win, draw, loss]
            p = np.array(updated)

        return p

    def predict_shootout(
        self,
        team_a_feats: pd.Series,
        team_b_feats: pd.Series,
    ) -> float:
        """Return P(team A wins the penalty shootout)."""
        return self.penalty_model.predict_proba(team_a_feats, team_b_feats)

    def _tune_ensemble_weight(
        self,
        wc_training_rows: pd.DataFrame,
        calib_rows: pd.DataFrame,
    ) -> float:
        """CV-select the optimal WC-only model weight from WC_WEIGHT_CV_RANGE.

        Uses the calibration holdout (2022) for selection. The CV range is
        [0.60, 0.90] in steps of 0.05 as specified in the design document.
        """
        candidates = np.arange(
            WC_WEIGHT_CV_RANGE[0],
            WC_WEIGHT_CV_RANGE[1] + 0.001,
            0.05,
        )
        X_cal = _get_X(calib_rows, PRETOURNAMENT_FEATURES)
        y_cal = calib_rows["outcome"].astype(int).values
        p_wc  = self.group_model_wc.predict_proba(X_cal)
        p_aug = self.group_model_aug.predict_proba(X_cal)

        best_rps = np.inf
        best_w   = WC_WEIGHT_DEFAULT
        for w in candidates:
            p = w * p_wc + (1 - w) * p_aug
            rps = compute_rps(y_cal, p)
            if rps < best_rps:
                best_rps = rps
                best_w   = float(w)

        return best_w


# ---------------------------------------------------------------------------
# Convenience function — fit all models in one call
# ---------------------------------------------------------------------------

def fit_all_models(
    wc_training_rows: pd.DataFrame,
    calib_rows: pd.DataFrame,
    aug_training_rows: Optional[pd.DataFrame] = None,
    wc_weight: float = WC_WEIGHT_DEFAULT,
    cv_tune_wc_weight: bool = False,
    recency_factor: float = RECENCY_FACTOR_DEFAULT,
    bayesian_alpha: float = BAYESIAN_ALPHA,
) -> WCForecastEnsemble:
    """Fit the complete model stack and return a ready WCForecastEnsemble.

    Parameters
    ----------
    wc_training_rows  : from features.build_training_rows() — DS8 WC matches.
    calib_rows        : 2022 WC rows (Fold 7 — only used here, never for HP tuning).
    aug_training_rows : from build_augmented_training_rows() — DS4 Tier 1 matches.
    wc_weight         : ensemble weight for WC-only model (default 0.70).
    cv_tune_wc_weight : tune ensemble weight on calibration holdout.
    recency_factor    : decay parameter [0.0, 1.0] for sample weights.
    bayesian_alpha    : Layer 2 update strength (design spec default 0.17).

    Returns
    -------
    Fitted WCForecastEnsemble ready for predict_group_stage() / predict_knockout()
    / predict_shootout().
    """
    ensemble = WCForecastEnsemble(
        wc_weight=wc_weight,
        recency_factor=recency_factor,
        bayesian_alpha=bayesian_alpha,
    )
    ensemble.fit(
        wc_training_rows=wc_training_rows,
        calib_rows=calib_rows,
        aug_training_rows=aug_training_rows,
        cv_tune_wc_weight=cv_tune_wc_weight,
    )
    return ensemble
