"""Check that the numbers printed in the manuscript still match the data.

Every figure the manuscript reports is derived from the CSVs in this
repository, so the two can drift apart whenever the analysis is re-run. They
have: at various points the text reported a question-answer count, a pair of
standard deviations, an inter-rater coefficient and a set of regression
coefficients that the data no longer supported.

This script recomputes each reported quantity, formats it the way the
manuscript does, and checks that the resulting string occurs in main.tex. It
is a guard against stale numbers, not a test of the analysis: a check passes
when the paper agrees with the data, whether or not either is correct.

    python verify_paper.py          # exits non-zero if anything has drifted

Numbers that do not appear verbatim are recomputed anyway and reported, so a
reader can see what the check could not pin down.
"""

from __future__ import annotations

import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
TEX = (ROOT / "paper" / "main.tex").read_text()

# Whitespace in the source is not meaningful for these checks: a number may
# sit either side of a line break that LaTeX will discard.
FLAT = " ".join(TEX.split())


def holm(pvals: np.ndarray) -> np.ndarray:
    order = np.argsort(pvals)
    m = len(pvals)
    out = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, pvals[idx] * (m - rank))
        out[idx] = min(running, 1.0)
    return out


def krippendorff_alpha_ordinal(units: np.ndarray) -> float:
    """Krippendorff's alpha with the ordinal difference function."""
    units = np.asarray(units, dtype=int)
    values = np.unique(units)
    index = {v: i for i, v in enumerate(values)}
    marginal = np.zeros(len(values))
    for value in units.ravel():
        marginal[index[value]] += 1
    n_total = marginal.sum()
    delta = np.zeros((len(values), len(values)))
    for c_ in range(len(values)):
        for k in range(c_, len(values)):
            span = marginal[c_ : k + 1].sum() - (marginal[c_] + marginal[k]) / 2
            delta[c_, k] = delta[k, c_] = span**2
    d_obs = (
        sum(
            sum(delta[index[x], index[y]] for x in row for y in row) / (len(row) - 1)
            for row in units
        )
        / n_total
    )
    d_exp = sum(
        marginal[a] * marginal[b] * delta[a, b]
        for a in range(len(values))
        for b in range(len(values))
    ) / (n_total * (n_total - 1))
    return 1 - d_obs / d_exp


def table_rows(label: str) -> dict[str, list[str]]:
    """Return the body rows of the table carrying \\label{label}.

    Keyed by the row's first cell, with the remaining cells as written. A
    whole-row lookup is what makes a check meaningful: searching the file for
    a bare "0.008" passes as soon as that string occurs anywhere, and in this
    manuscript it occurs in an adjusted R-squared as well as in a p-value.
    """
    start = TEX.index(f"\\label{{{label}}}")
    block = TEX[start : TEX.index("\\end{tabularx}", start)]
    rows: dict[str, list[str]] = {}
    for line in block.split("\\\\"):
        line = " ".join(line.split())
        if "&" not in line or "multicolumn" in line:
            continue
        cells = [cell.strip() for cell in line.split("&")]
        key = re.sub(r"\\[a-zA-Z]+\*?(\{[^}]*\})?|[${}]", "", cells[0]).strip()
        if key:
            rows.setdefault(key, cells[1:])
    return rows


def clean(cell: str) -> str:
    """Strip the maths and markup LaTeX wraps a number in."""
    return (
        re.sub(r"\\[a-zA-Z]+|[${}\\]|\^\{?\*+\}?", "", cell).replace("-", "-").strip()
    )


class Checker:
    def __init__(self) -> None:
        self.passed = 0
        self.failures: list[tuple[str, str]] = []

    def expect(self, label: str, text: str) -> None:
        """Assert that `text` occurs in the manuscript."""
        if " ".join(text.split()) in FLAT:
            self.passed += 1
        else:
            self.failures.append((label, text))

    def cell(self, label: str, actual: str, expected: str) -> None:
        """Assert that a specific table cell holds the expected value."""
        if clean(actual) == expected:
            self.passed += 1
        else:
            self.failures.append((label, f"{expected} (table has {clean(actual)!r})"))

    def report(self) -> int:
        print(f"\n{self.passed} checks passed, {len(self.failures)} failed")
        for label, text in self.failures:
            print(f"\n  FAILED  {label}")
            print(f"    not found in main.tex: {text!r}")
        return 1 if self.failures else 0


def main() -> int:
    c = Checker()
    agg = pd.read_csv(DATA / "agg.csv")
    quality = pd.read_csv(DATA / "relevance_and_specificity_scores.csv")
    pairs = pd.read_csv(DATA / "pair_level_scores.csv")
    coders = pd.read_csv(DATA / "coder_agreement.csv")
    df = agg.merge(
        quality[["conversation_id", "specificity", "relevance"]], on="conversation_id"
    )
    ai = df["Interviewer"] == "AI"

    # ---- sample and corpus sizes -------------------------------------
    c.expect("N interviews", f"$N = {len(df)}$")
    c.expect("analysed answers", f"{agg['question_count'].sum()} question-answer pairs")
    for cond, n in agg.groupby("Interviewer")["question_count"].sum().items():
        c.expect(f"answers, {cond}", str(n))
    c.expect("coded pairs", f"{len(pairs)} of these")
    for cond, n in pairs.groupby("interviewer").size().items():
        c.expect(f"coded pairs, {cond}", str(n))

    # ---- inter-rater reliability -------------------------------------
    from sklearn.metrics import cohen_kappa_score

    irr = table_rows("tab:irr")
    for measure in ("Specificity", "Relevance"):
        a = coders[f"{measure.lower()}_coder_a"].values.astype(int)
        b = coders[f"{measure.lower()}_coder_b"].values.astype(int)
        exact, within = (a == b).mean(), (abs(a - b) <= 1).mean()
        alpha = krippendorff_alpha_ordinal(np.column_stack([a, b]))
        kappa = cohen_kappa_score(a, b, weights="quadratic")
        cells = irr[measure]
        c.cell(f"{measure}: exact agreement", cells[0], f"{exact:.3f}")
        c.cell(f"{measure}: within one point", cells[1], f"{within:.3f}")
        c.cell(f"{measure}: Krippendorff alpha", cells[2], f"{alpha:.3f}")
        c.cell(f"{measure}: Cohen kappa", cells[3], f"{kappa:.3f}")
        c.expect(
            f"{measure}: exact agreement in prose", f"{round(exact * 100)} percent"
        )

    # ---- the condition comparison ------------------------------------
    outcomes = [
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
    raw, stats = [], []
    for label, col in outcomes:
        x, y = df.loc[ai, col].dropna(), df.loc[~ai, col].dropna()
        u, p = mannwhitneyu(x, y, alternative="two-sided")
        cles = u / (len(x) * len(y))
        raw.append(p)
        stats.append((label, u, 2 * cles - 1, cles))
    adjusted = holm(np.array(raw))

    bonferroni = np.clip(np.array(raw) * len(raw), None, 1.0)
    mwu_rows = table_rows("tab:mwu")
    for (label, u, rbc, cles), p, pb in zip(stats, adjusted, bonferroni):
        cells = mwu_rows[label]
        c.cell(f"{label}: U", cells[0], f"{u:.1f}")
        c.cell(f"{label}: Holm p", cells[1], "<0.001" if p < 0.001 else f"{p:.3f}")
        c.cell(
            f"{label}: Bonferroni p", cells[2], "<0.001" if pb < 0.001 else f"{pb:.3f}"
        )
        c.cell(f"{label}: RBC", cells[3], f"{rbc:.3f}")
        c.cell(f"{label}: CLES", cells[4], f"{cles:.3f}")

    # ---- descriptive values quoted in the Results --------------------
    for col, fmt, label in [
        ("avg_response_length", "{:.0f}", "mean response length"),
        ("characters_per_min", "{:.0f}", "characters per minute"),
        ("avg_response_duration_by_interviewer(sec)", "{:.0f}", "formulation time"),
    ]:
        for cond, mask in (("AI", ai), ("Human", ~ai)):
            c.expect(f"{label}, {cond}", fmt.format(df.loc[mask, col].mean()))

    c.expect("median duration, AI", f"{df.loc[ai, 'duration(min)'].median():.1f}")
    c.expect("median duration, Human", f"{df.loc[~ai, 'duration(min)'].median():.1f}")
    c.expect("mean duration, AI", f"{df.loc[ai, 'duration(min)'].mean():.1f}")
    c.expect("mean duration, Human", f"{df.loc[~ai, 'duration(min)'].mean():.1f}")

    # Totals are printed with a thousands separator.
    for cond, mask in (("AI", ai), ("Human", ~ai)):
        total = df.loc[mask, "total_character_by_user"].mean()
        c.expect(f"interview length, {cond}", f"{total:,.0f}")

    # ---- regression estimates ----------------------------------------
    import statsmodels.api as sm

    day = pd.read_csv(DATA / "interview_day.csv")
    reg = df.merge(day, on="conversation_id")
    reg["ai"] = reg["Interviewer"].eq("AI").astype(float)
    # Table 5 prints the two score outcomes to three decimals and the rest to
    # two, and writes a negative sign inside the maths rather than beside it.
    for label, col, dp in [
        ("Interview length", "total_character_by_user", 2),
        ("Response length", "avg_response_length", 2),
        ("Question count", "question_count", 2),
        ("Specificity", "specificity", 3),
        ("Relevance", "relevance", 3),
    ]:
        for fe in (False, True):
            X = reg[["ai"]]
            if fe:
                X = X.join(
                    pd.get_dummies(reg["interview_day"], prefix="d", drop_first=True)
                )
            model = sm.OLS(reg[col], sm.add_constant(X).astype(float)).fit()
            name = f"{label} coefficient{' with day FE' if fe else ''}"
            c.expect(name, f"${model.params['ai']:.{dp}f}")
            c.expect(f"{name}, SE", f"$({model.bse['ai']:.{dp}f})$")
            c.expect(f"{name}, R2", f"{model.rsquared:.3f}")

    # ---- equivalence bounds ------------------------------------------
    # The intervals are printed in the text, not in a table, so they are
    # matched together with their surrounding "to" to pin each pair down.
    from stats_tables import FAMILY_ALPHA, hodges_lehmann, robustness

    level = FAMILY_ALPHA / len(outcomes)
    later = reg[reg["interview_day"] != 1]
    later_ai = later["Interviewer"] == "AI"
    # Bounds are differences of score means, so some fall exactly on a half
    # (0.045); round those half-up as the manuscript does, not to the binary
    # float just below.
    def half_up(value: float, places: int, sign: bool) -> str:
        q = Decimal(str(round(value, 9))).quantize(
            Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP
        )
        return f"+{q}" if sign and q > 0 else str(q)

    for name, frame, mask, col, places, sign in [
        ("duration interval", reg, ai, "duration(min)", 1, True),
        ("specificity interval", reg, ai, "specificity", 2, False),
        ("relevance interval", reg, ai, "relevance", 2, False),
        ("specificity interval, no day 1", later, later_ai, "specificity", 2, False),
        ("relevance interval, no day 1", later, later_ai, "relevance", 2, False),
    ]:
        _, lo, hi = hodges_lehmann(
            frame.loc[mask, col].values, frame.loc[~mask, col].values, level
        )
        c.expect(
            name, f"${half_up(lo, places, sign)}$ to ${half_up(hi, places, sign)}$"
        )

    # ---- robustness to the day structure (Appendix C) ----------------
    fmt_p = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
    if "\\label{tab:robust}" not in TEX:
        c.failures.append(("robustness table", "\\label{tab:robust}"))
    else:
        robust_rows = table_rows("tab:robust")
        for _, r in robustness(reg).iterrows():
            cells = robust_rows[r["outcome"]]
            c.cell(f"{r['outcome']}: full sample", cells[0], fmt_p(r["full"]))
            c.cell(f"{r['outcome']}: within days", cells[1], fmt_p(r["within_day"]))
            c.cell(
                f"{r['outcome']}: without day 1", cells[2], fmt_p(r["without_day_1"])
            )
    c.expect("interviews without day 1", f"remaining {len(later)} interviews")
    both = reg.groupby("interview_day")["ai"].transform("nunique") == 2
    c.expect("interviews on mixed days", f"the {both.sum()} interviews")
    human = reg[reg["ai"] == 0]
    first = human["interview_day"] == 1
    # The manuscript spells out counts below ten.
    words = "zero one two three four five six seven eight nine".split()
    c.expect("day-1 human interviews", f"Its {words[first.sum()]} human-led interviews")
    c.expect(
        "day-1 retained", f"The {words[first.sum()]} human-led interviews from that day"
    )
    c.expect("later human interviews", f"the {(~first).sum()} human-led interviews")
    for col, fmt in [("avg_response_length", "{:.0f}"), ("specificity", "{:.2f}")]:
        c.expect(
            f"{col}, day 1 against later",
            f"{fmt.format(human.loc[first, col].mean())} against "
            f"{fmt.format(human.loc[~first, col].mean())}",
        )

    return c.report()


if __name__ == "__main__":
    sys.exit(main())
