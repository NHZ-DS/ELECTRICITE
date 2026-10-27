import pandas as pd

from gridcast.features import FEATURES_ML, build_features


def test_no_feature_uses_the_future(synthetic):
    """Day-ahead rule: changing today's consumption must not change any feature of today."""
    day = pd.Timestamp("2024-02-10", tz="Europe/Paris")
    base = build_features(synthetic)
    altered = synthetic.copy()
    altered.loc[altered["ts"] >= day, "load_mw"] *= 2
    after = build_features(altered)
    today = (base["ts"] >= day) & (base["ts"] < day + pd.Timedelta(days=1))
    pd.testing.assert_frame_equal(base.loc[today, FEATURES_ML], after.loc[today, FEATURES_ML])


def test_holidays_and_day_type(synthetic):
    f = build_features(synthetic)
    christmas = f[f["ts"].dt.strftime("%m-%d") == "12-25"]
    assert christmas["is_holiday"].eq(1).all()
    assert christmas["day_type"].eq(2).all()
