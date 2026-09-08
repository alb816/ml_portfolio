from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ab_test import generate_data, run_ab_test


def test_generate_data_shape_and_columns():
    df = generate_data(seed=7, n_a=50, n_b=60)

    assert list(df.columns) == ["group", "revenue"]
    assert len(df) == 110
    assert df["group"].nunique() == 2


def test_run_ab_test_detects_significant_difference():
    df = generate_data(
        seed=42,
        n_a=2000,
        n_b=2000,
        mean_a=10.0,
        mean_b=11.0,
        std_a=2.0,
        std_b=2.0,
    )

    result = run_ab_test(df)

    assert result["significant"] is True
    assert result["abs_lift"] > 0
    assert result["p_value"] < 0.05


def test_run_ab_test_detects_no_significant_difference():
    values = np.full(2000, 10.0)
    df = pd.DataFrame(
        {
            "group": ["A"] * 2000 + ["B"] * 2000,
            "revenue": np.concatenate([values, values]),
        }
    )

    result = run_ab_test(df)

    assert result["significant"] is False
    assert abs(result["abs_lift"]) == 0.0
    assert result["p_value"] == 1.0
