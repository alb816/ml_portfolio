import numpy as np
import pandas as pd
from scipy import stats


CONTROL_GROUP = "A"
TREATMENT_GROUP = "B"
METRIC_COLUMN = "revenue"
GROUP_COLUMN = "group"


def generate_data(
    seed: int = 42,
    n_a: int = 1000,
    n_b: int = 1000,
    mean_a: float = 10.0,
    mean_b: float = 10.2,
    std_a: float = 2.0,
    std_b: float = 2.0,
) -> pd.DataFrame:
    """Generate synthetic data for A/B test with two groups."""
    rng = np.random.default_rng(seed)

    group_a = rng.normal(loc=mean_a, scale=std_a, size=n_a)
    group_b = rng.normal(loc=mean_b, scale=std_b, size=n_b)

    return pd.DataFrame(
        {
            GROUP_COLUMN: [CONTROL_GROUP] * n_a + [TREATMENT_GROUP] * n_b,
            METRIC_COLUMN: np.concatenate([group_a, group_b]),
        }
    )


def _welch_confidence_interval(a: np.ndarray, b: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    diff = mean_b - mean_a

    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    se = np.sqrt(var_a / len(a) + var_b / len(b))

    df_num = (var_a / len(a) + var_b / len(b)) ** 2
    df_den = (var_a / len(a)) ** 2 / (len(a) - 1) + (var_b / len(b)) ** 2 / (len(b) - 1)
    df = df_num / df_den

    t_crit = stats.t.ppf((1 + confidence) / 2, df=df)
    margin = t_crit * se
    return diff - margin, diff + margin


def run_ab_test(
    df: pd.DataFrame,
    metric_col: str = METRIC_COLUMN,
    group_col: str = GROUP_COLUMN,
    control_group: str = CONTROL_GROUP,
    treatment_group: str = TREATMENT_GROUP,
) -> dict:
    """Run Welch's t-test and summarize the experiment result."""
    if df.empty:
        raise ValueError("Input data frame is empty.")

    if metric_col not in df.columns or group_col not in df.columns:
        raise ValueError(f"Columns '{metric_col}' and '{group_col}' must be present in the data frame.")

    if control_group not in df[group_col].unique() or treatment_group not in df[group_col].unique():
        raise ValueError("Both control and treatment groups must exist in the dataset.")

    control = df.loc[df[group_col] == control_group, metric_col].to_numpy(dtype=float)
    treatment = df.loc[df[group_col] == treatment_group, metric_col].to_numpy(dtype=float)

    if len(control) < 2 or len(treatment) < 2:
        raise ValueError("Each group must contain at least two observations.")

    mean_a = float(np.mean(control))
    mean_b = float(np.mean(treatment))
    abs_lift = mean_b - mean_a
    rel_lift = abs_lift / mean_a if mean_a else 0.0

    if np.allclose(control, treatment):
        return {
            "mean_a": mean_a,
            "mean_b": mean_b,
            "abs_lift": abs_lift,
            "relative_lift": rel_lift,
            "t_statistic": 0.0,
            "p_value": 1.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "significant": False,
            "control_group": control_group,
            "treatment_group": treatment_group,
            "n_control": len(control),
            "n_treatment": len(treatment),
        }

    result = stats.ttest_ind(treatment, control, equal_var=False, alternative="two-sided")
    t_stat = float(result.statistic)
    p_value = float(result.pvalue)
    if not np.isfinite(p_value):
        p_value = 1.0
        t_stat = 0.0

    ci_lower, ci_upper = _welch_confidence_interval(control, treatment)

    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "abs_lift": abs_lift,
        "relative_lift": rel_lift,
        "t_statistic": t_stat,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": p_value < 0.05,
        "control_group": control_group,
        "treatment_group": treatment_group,
        "n_control": len(control),
        "n_treatment": len(treatment),
    }


def main() -> None:
    df = generate_data(seed=42, mean_a=10.0, mean_b=10.2)
    result = run_ab_test(df)

    print("Результат A/B теста")
    print("-" * 40)
    print(f"Среднее A: {result['mean_a']:.4f}")
    print(f"Среднее B: {result['mean_b']:.4f}")
    print(f"Абсолютный эффект: {result['abs_lift']:.4f}")
    print(f"Относительный эффект: {result['relative_lift'] * 100:.2f}%")
    print(f"Статистика Welch t: {result['t_statistic']:.4f}")
    print(f"p-value: {result['p_value']:.6f}")
    print(f"95% доверительный интервал: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")

    if result["significant"]:
        direction = "выигрывает" if result["abs_lift"] > 0 else "проигрывает"
        print(f"Решение: эффект статистически значим, версия B {direction} по сравнению с A.")
        print(
            "Бизнесовый вывод: изменение выглядит полезным, но важно проверить, "
            "что это не случайный шум и что эффект устойчив на реальных данных."
        )
    else:
        print("Решение: статистически значимых различий не обнаружено.")
        print(
            "Бизнесовый вывод: пока нет оснований считать, что изменение влияет на метрику. "
            "Нужны дополнительные данные или более чувствительный эксперимент."
        )


if __name__ == "__main__":
    main()
