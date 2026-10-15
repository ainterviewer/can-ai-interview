"""Compute the Mann-Whitney table and the equivalence bounds for the paper.

Reads the files written by build_data.py and prints Table 3 (the
condition comparison), the Hodges-Lehmann shift estimates, and the
robustness checks for the day structure reported in Appendix C.

Multiplicity is controlled with Holm's step-down procedure. Holm holds the
family-wise error rate at the same level as Bonferroni under no additional
assumptions and is uniformly more powerful, so it is reported as primary;
the Bonferroni column is kept alongside it for comparison. Several of these
outcomes are algebraic functions of the same two quantities, which makes
single-step Bonferroni particularly conservative here.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm, rankdata

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
FAMILY_ALPHA = 0.05

OUTCOMES = [
    ("Duration", "duration(min)"),
    ("Response length", "avg_response_length"),
    ("Interview length", "total_character_by_user"),
    ("Questions per interview", "question_count"),
    ("Questions per minute", "questions_per_min"),
    ("Characters per minute", "characters_per_min"),
    ("Question formulation time", "avg_response_duration_by_interviewer(sec)"),
    ("Mean specificity", "specificity"),
    ("Mean relevance", "relevance"),
]


def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm-adjusted p-values, returned in the order given."""
    order = np.argsort(pvals)
    m = len(pvals)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, pvals[idx] * (m - rank))
        adjusted[idx] = min(running, 1.0)
    return adjusted


def effect_sizes(ai: np.ndarray, human: np.ndarray, u: float) -> tuple[float, float]:
    """Rank-biserial correlation and common-language effect size.

    Both are oriented so that positive favours the AI condition.
    """
    cles = u / (len(ai) * len(human))
    return 2 * cles - 1, cles


def hodges_lehmann(ai: np.ndarray, human: np.ndarray, alpha: float):
    """HL shift estimate with a distribution-free CI at level `alpha`.

    The estimate is the median of all pairwise differences; the interval is
    the corresponding order statistics of those differences, which is the
    interval that inverts the Mann-Whitney test.
    """
    diffs = np.sort([a - h for a, h in itertools.product(ai, human)])
    n = len(diffs)
    estimate = float(np.median(diffs))
    # Normal approximation to the Wilcoxon rank-sum critical value.
    na, nh = len(ai), len(human)
    z = norm.ppf(1 - alpha / 2)
    spread = z * np.sqrt(na * nh * (na + nh + 1) / 12.0)
    k = int(np.floor(na * nh / 2.0 - spread))
    lower = diffs[max(k, 0)]
    upper = diffs[min(n - 1 - max(k, 0), n - 1)]
    return estimate, float(lower), float(upper)


def van_elteren(values: np.ndarray, is_ai: np.ndarray, strata: np.ndarray) -> float:
    """Two-sided p-value of the van Elteren test, the stratified rank-sum test.

    Each stratum contributes its AI rank sum minus its null expectation,
    weighted by 1 / (n + 1); the tie-corrected variances are combined with the
    squared weights. A stratum holding only one condition carries no
    information about the difference and contributes nothing.
    """
    stat = var = 0.0
    for s in np.unique(strata):
        x, a = values[strata == s], is_ai[strata == s]
        m, n = a.sum(), (~a).sum()
        if m == 0 or n == 0:
            continue
        size = m + n
        weight = 1.0 / (size + 1)
        _, ties = np.unique(x, return_counts=True)
        v = m * n / 12.0 * ((size + 1) - (ties**3 - ties).sum() / (size * (size - 1)))
        stat += weight * (rankdata(x)[a].sum() - m * (size + 1) / 2.0)
        var += weight**2 * v
    return float(2 * norm.sf(abs(stat / np.sqrt(var))))


def load() -> pd.DataFrame:
    """One row per interview: behaviour, coded quality and day index."""
    agg = pd.read_csv(DATA / "agg.csv")
    quality = pd.read_csv(DATA / "relevance_and_specificity_scores.csv")
    day = pd.read_csv(DATA / "interview_day.csv")
    return agg.merge(
        quality[["conversation_id", "specificity", "relevance"]],
        on="conversation_id",
    ).merge(day, on="conversation_id")


def robustness(df: pd.DataFrame) -> pd.DataFrame:
    """Holm-adjusted p-values for Table 6, one row per outcome.

    The first day of data collection ran the human condition only, so the
    pooled comparison is repeated within days (van Elteren) and with that day
    removed (Mann-Whitney on the remaining interviews).
    """
    is_ai = (df["Interviewer"] == "AI").values
    later = df[df["interview_day"] != 1]
    later_ai = later["Interviewer"] == "AI"
    pooled, within, dropped = [], [], []
    for _, column in OUTCOMES:
        pooled.append(mannwhitneyu(df.loc[is_ai, column], df.loc[~is_ai, column])[1])
        within.append(van_elteren(df[column].values, is_ai, df["interview_day"].values))
        dropped.append(
            mannwhitneyu(later.loc[later_ai, column], later.loc[~later_ai, column])[1]
        )
    return pd.DataFrame(
        {
            "outcome": [label for label, _ in OUTCOMES],
            "full": holm(np.array(pooled)),
            "within_day": holm(np.array(within)),
            "without_day_1": holm(np.array(dropped)),
        }
    )


def main() -> None:
    df = load()

    rows = []
    for label, column in OUTCOMES:
        ai = df.loc[df["Interviewer"] == "AI", column].dropna().values
        human = df.loc[df["Interviewer"] == "Human", column].dropna().values
        u, p = mannwhitneyu(ai, human, alternative="two-sided")
        rbc, cles = effect_sizes(ai, human, u)
        rows.append(
            {
                "outcome": label,
                "AI": ai.mean(),
                "Human": human.mean(),
                "U": u,
                "raw": p,
                "RBC": rbc,
                "CLES": cles,
            }
        )

    table = pd.DataFrame(rows)
    m = len(table)
    table["bonferroni"] = (table["raw"] * m).clip(upper=1.0)
    table["holm"] = holm(table["raw"].values)

    print(f"Table 3  (N = {len(df)}, {m} tests)\n")
    header = (
        f"{'outcome':26}{'AI':>9}{'Human':>10}{'U':>8}"
        f"{'raw p':>9}{'Holm':>8}{'Bonf':>8}{'RBC':>8}{'CLES':>7}"
    )
    print(header)
    print("-" * len(header))
    for _, r in table.iterrows():
        print(
            f"{r['outcome']:26}{r['AI']:9.2f}{r['Human']:10.2f}{r['U']:8.1f}"
            f"{r['raw']:9.4f}{r['holm']:8.3f}{r['bonferroni']:8.3f}"
            f"{r['RBC']:8.3f}{r['CLES']:7.3f}"
        )

    print(f"\nHodges-Lehmann shift, AI minus human, CI at family level {FAMILY_ALPHA}")
    for label, column in [
        ("Duration", "duration(min)"),
        ("Mean specificity", "specificity"),
        ("Mean relevance", "relevance"),
    ]:
        ai = df.loc[df["Interviewer"] == "AI", column].dropna().values
        human = df.loc[df["Interviewer"] == "Human", column].dropna().values
        est, lo, hi = hodges_lehmann(ai, human, FAMILY_ALPHA / m)
        print(f"  {label:20} shift {est:+.3f}   CI [{lo:+.3f}, {hi:+.3f}]")

    later = df[df["interview_day"] != 1]
    print(f"\nTable 6  Holm-adjusted p, day structure (without day 1: N = {len(later)})\n")
    print(f"{'outcome':26}{'full':>8}{'within':>8}{'no day 1':>10}")
    for _, r in robustness(df).iterrows():
        print(
            f"{r['outcome']:26}{r['full']:8.3f}{r['within_day']:8.3f}"
            f"{r['without_day_1']:10.3f}"
        )

    print("\nHodges-Lehmann shift without day 1, CI at family level")
    for label, column in [("Mean specificity", "specificity"), ("Mean relevance", "relevance")]:
        ai = later.loc[later["Interviewer"] == "AI", column].values
        human = later.loc[later["Interviewer"] == "Human", column].values
        est, lo, hi = hodges_lehmann(ai, human, FAMILY_ALPHA / m)
        print(f"  {label:20} shift {est:+.3f}   CI [{lo:+.3f}, {hi:+.3f}]")

    human = df[df["Interviewer"] == "Human"]
    print("\nHuman-led interviews, day 1 against later days")
    for column in ["avg_response_length", "specificity"]:
        first = human.loc[human["interview_day"] == 1, column].mean()
        rest = human.loc[human["interview_day"] != 1, column].mean()
        print(f"  {column:22} day 1 {first:7.2f}   later {rest:7.2f}")


if __name__ == "__main__":
    main()
