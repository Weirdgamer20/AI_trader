import pytest
import numpy as np
import pandas as pd
from src.ledger.database import DatabaseManager
from src.meta_learning.meta_features import MetaFeatureExtractor
from src.meta_learning.meta_selector import MetaLearnerSelector
from src.meta_learning.experiment_registry import ExperimentRegistry
from src.meta_learning.critic import IndependentStrategyCritic

def test_meta_features_extraction():
    extractor = MetaFeatureExtractor()
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=40, freq="15min", tz="UTC"),
        "open": np.linspace(100, 110, 40),
        "high": np.linspace(102, 112, 40),
        "low": np.linspace(99, 109, 40),
        "close": np.linspace(101, 111, 40),
        "volume": np.random.uniform(5, 10, 40),
    })
    meta = extractor.extract_meta_features(df, {"m1": 0.58, "m2": 0.54})
    assert "volatility_regime" in meta
    assert "trend_strength" in meta
    assert meta["trend_strength"] > 0 # Bullish upward sequence
    assert "learner_consensus" in meta
    assert meta["learner_consensus"] == 0.56

def test_meta_learner_selector_regimes():
    selector = MetaLearnerSelector()

    # High volatility regime
    cfg1 = selector.select_optimal_configuration(
        {"volatility_regime": 0.025, "trend_strength": 0.0}, ["HIGH_VOLATILITY"]
    )
    assert cfg1["selected_algorithm"] == "hist_gradient_boosting"
    assert cfg1["selected_horizon_candles"] == 1

    # Bullish trending regime
    cfg2 = selector.select_optimal_configuration(
        {"volatility_regime": 0.008, "trend_strength": 0.008}, ["TRENDING_BULLISH"]
    )
    assert cfg2["selected_horizon_candles"] == 2

    # Range bound
    cfg3 = selector.select_optimal_configuration(
        {"volatility_regime": 0.005, "trend_strength": 0.0002}, ["RANGE_BOUND"]
    )
    assert cfg3["selected_algorithm"] == "logistic_regression"

def test_experiment_registry_and_rollback(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "meta_test.db")
    reg = ExperimentRegistry(db=db, registry_dir=tmp_path / "registry")

    champ = reg.active_champion
    assert champ["model_id"] == "champ-baseline-hgb-v1"

    # Inferior challenger: should NOT promote
    bad_res = reg.register_challenger("bad-challenger", "random_forest", 1, val_accuracy=0.51, val_mae=1600.0)
    assert bad_res["promoted_to_champion"] is False
    assert reg.active_champion["model_id"] == "champ-baseline-hgb-v1"

    # Superior challenger: should promote
    good_res = reg.register_challenger("good-challenger", "hist_gradient_boosting", 1, val_accuracy=0.56, val_mae=1400.0)
    assert good_res["promoted_to_champion"] is True
    assert reg.active_champion["model_id"] == "good-challenger"

    # Rollback to baseline
    rb_res = reg.rollback_to_previous_champion()
    assert rb_res["success"] is True
    assert reg.active_champion["model_id"] == "champ-baseline-hgb-v1"

def test_independent_strategy_critic():
    critic = IndependentStrategyCritic()

    # Good proposal
    proj_good = {
        "direction": "UP",
        "probability_up": 0.62,
        "expected_return_pct": 0.025, # 2.5% expected return
        "uncertainty_width_pct": 0.03, # 3% uncertainty width
    }
    eval1 = critic.evaluate_proposal("I-BTC_INR", "BUY", proj_good)
    assert eval1["approved"] is True
    assert eval1["critic_score"] == 1.0

    # High uncertainty proposal: Critic must reject
    proj_uncertain = dict(proj_good, uncertainty_width_pct=0.08)
    eval2 = critic.evaluate_proposal("I-BTC_INR", "BUY", proj_uncertain)
    assert eval2["approved"] is False
    assert any("uncertainty" in r for r in eval2["rejection_reasons"])

    # Weak return proposal (cannot clear fees): Critic must reject
    proj_low_return = dict(proj_good, expected_return_pct=0.005)
    eval3 = critic.evaluate_proposal("I-BTC_INR", "BUY", proj_low_return)
    assert eval3["approved"] is False
    assert any("cost hurdle" in r for r in eval3["rejection_reasons"])

    # Disagreeing consensus
    eval4 = critic.evaluate_proposal("I-BTC_INR", "BUY", proj_good, model_agreement_pct=0.50)
    assert eval4["approved"] is False
    assert any("disagreement" in r for r in eval4["rejection_reasons"])
