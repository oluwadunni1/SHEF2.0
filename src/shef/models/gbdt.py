"""
Tabular model builders (defaults) and Optuna search spaces (from v1 hpo.py).
Thread counts are capped at the machine's 4 cores.
"""
from __future__ import annotations

N_JOBS = 4
SEED = 42


def ridge(alpha: float = 1.0):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # Linear models need imputed, scaled inputs; trees below take NaN natively.
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))


def lgbm(**params):
    import lightgbm as lgb

    base = dict(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, random_state=SEED, n_jobs=N_JOBS, verbosity=-1)
    return lgb.LGBMRegressor(**{**base, **params})


def xgb(**params):
    import xgboost

    base = dict(n_estimators=600, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8,
                tree_method="hist", random_state=SEED, n_jobs=N_JOBS)
    return xgboost.XGBRegressor(**{**base, **params})


def catboost(**params):
    import catboost as cb

    base = dict(iterations=800, learning_rate=0.05, depth=8, random_seed=SEED, verbose=0, thread_count=N_JOBS,
                allow_writing_files=False)
    return cb.CatBoostRegressor(**{**base, **params})


BUILDERS = {"ridge": ridge, "lgbm": lgbm, "xgb": xgb, "catboost": catboost}


def search_space(name: str, trial) -> dict:
    """v1's Optuna spaces (src/experiments/hpo.py)."""
    if name == "xgb":
        return dict(n_estimators=trial.suggest_int("n_estimators", 200, 800),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                    max_depth=trial.suggest_int("max_depth", 3, 10),
                    min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True))
    if name == "lgbm":
        return dict(n_estimators=trial.suggest_int("n_estimators", 200, 800),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                    num_leaves=trial.suggest_int("num_leaves", 16, 128),
                    max_depth=trial.suggest_int("max_depth", 3, 10),
                    min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True))
    if name == "catboost":
        return dict(iterations=trial.suggest_int("iterations", 200, 800),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                    depth=trial.suggest_int("depth", 3, 10),
                    l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-2, 10, log=True),
                    bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
                    random_strength=trial.suggest_float("random_strength", 1e-3, 10, log=True),
                    border_count=trial.suggest_int("border_count", 32, 255))
    raise ValueError(name)
