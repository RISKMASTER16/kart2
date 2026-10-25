"""Registry of predictors. backtest.py runs everything listed in PREDICTORS.

To add a model: write predict(history, now, horizon=1) -> pd.Series in its own file
(see INTERFACES.md section 3), import it here, and add one line to PREDICTORS.
"""
from pathlib import Path

from .baselines import last_24h_mean, last_hour, same_hour_yesterday
from .decay_predict import decay_predict

PREDICTORS = {
    "last_hour": last_hour,                       # standard rule, ~ "least recently used"
    "last_24h_mean": last_24h_mean,               # ~ "least frequently used"
    "same_hour_yesterday": same_hour_yesterday,   # daily rhythm
    "decay": decay_predict,                       # fitted attention-decay model
}

# XGBoost is only registered once it has been trained (python train_xgb.py), so the backtest
# keeps working before then.
if any((Path(__file__).resolve().parent.parent / "models").glob("xgb_h*.json")):
    from .xgb import xgb_predict
    PREDICTORS["xgboost"] = xgb_predict

# An experimental transformer predictor lives in future/transformer/ (see its README).
