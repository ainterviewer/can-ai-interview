"""Rebuild the public analysis files from the private source data.

The CSVs in this repository are derived. Until now they had no build script,
and one consequence was that 11 of the 400 coded question-answer pairs were
silently lost to a text join (see FILTER, note 4). This script makes the
derivation explicit and reproducible.

It reads two private sources that are NOT part of the public repository,
because they contain verbatim interview text and identifiable coder names:

  MESSAGES  message-level transcript export with Interviewer, created_at,
            character_count and word_count
  CODING    the two coders' independent score files, four blocks each

Both are read-only here. Nothing is written outside this directory, and no
verbatim text, respondent demographics or coder names enter the output.

FILTER
------
1. The third substantive case ("asked for help but did not get it") is
   dropped from every interview. It was administered on days 1-3 only and
   then dropped under time pressure because the human interviewers rarely
   reached it, so participants were not exposed to the same instrument.
2. The closing prompt is KEPT. It was administered in 18 of 20 AI and 20 of
   21 human interviews, so there is no comparability argument for cutting
   it, and keeping it avoids an exclusion that would rest on reconstructed
   section labels.
3. `<|skipquestion|>` control tokens are dropped from the answer counts.
   They are clicks on the interface skip button, not answers. Their
   incidence is reported separately as a behavioural measure.
4. Coded pairs are joined back to the transcript on answer text. Where the
   same answer text occurs more than once, the candidates are checked to
   confirm they fall in one conversation, which is the only level the join
   needs; otherwise the best fuzzy match above FUZZY_MIN is used. The join
   must be total -- the script asserts this rather than dropping rows.

Both filter boundaries are scripted questions that the human interviewers
reproduced near-verbatim, so they are located by text and need no section
labels. Guide-section assignment for the human interviews exists only as a
post-hoc reconstruction and is deliberately not used anywhere here.

DURATION
--------
Duration is the sum of the gaps between consecutive messages, restricted to
the analysed portion of the interview. Time spent in the dropped third case
is therefore excluded, so the rate measures put analysed answers over the
time in which they were collected. The previous agg.csv used an
undocumented definition that also removed the closing exchange and, for
human interviews only, a further 7-45 seconds; this script replaces it.
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"

THIRD_MARKER = "but did not get it"
CLOSE_MARKER = "before we conclude"
SKIP_TOKEN = "<|skipquestion|>"
FUZZY_MIN = 0.85


def squash(text: object) -> str:
    """Collapse runs of whitespace, for exact answer-text joins."""
    return " ".join(str(text).split())


def fold(text: object) -> str:
    """Lowercase and strip punctuation, for marker and fuzzy matching."""
    folded = str(text).lower().replace("’", "'")
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", folded).split())


def load_messages(messages: Path) -> pd.DataFrame:
    df = pd.read_excel(messages)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df = df.sort_values(["conversation_id", "created_at"]).reset_index(drop=True)
    df["gap_min"] = (
        df.groupby("conversation_id")["created_at"].diff().dt.total_seconds() / 60
    )
    df["gap_sec"] = df["gap_min"] * 60
    df["folded"] = df["content"].map(fold)
    df["zone"] = assign_zones(df)
    df["is_skip"] = df["content"].map(squash) == SKIP_TOKEN
    return df


def assign_zones(df: pd.DataFrame) -> pd.Series:
    """Label every message by the guide segment in progress when it was sent.

    The two markers are interviewer messages, so the zone changes on the
    question and carries forward over the answers that follow it.
    """
    zones = pd.Series("substantive", index=df.index, dtype=object)
    for _, block in df.groupby("conversation_id", sort=False):
        current = "substantive"
        for idx, row in block.iterrows():
            if row["role"] == "assistant":
                if THIRD_MARKER in row["folded"]:
                    current = "third_case"
                elif CLOSE_MARKER in row["folded"]:
                    current = "closing"
            zones[idx] = current
    return zones


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """One row per interview, over the analysed portion of each."""
    analysed = df[df["zone"] != "third_case"]
    rows = []
    for cid, block in analysed.groupby("conversation_id"):
        answers = block[(block["role"] == "user") & (~block["is_skip"])]
        probes = block[block["role"] == "assistant"]
        duration = block["gap_min"].sum()
        characters = answers["character_count"].sum()
        count = len(answers)
        rows.append(
            {
                "conversation_id": cid,
                "Interviewer": block["Interviewer"].iloc[0],
                "duration(min)": round(duration, 2),
                "question_count": count,
                "total_character_by_user": int(characters),
                "avg_response_length": round(characters / count, 2),
                "questions_per_min": round(count / duration, 3),
                "characters_per_min": round(characters / duration, 2),
                "avg_response_duration_by_interviewer(sec)": round(
                    probes["gap_sec"].mean(), 2
                ),
                "avg_response_duration_by_user(sec)": round(
                    answers["gap_sec"].mean(), 2
                ),
                "skip_clicks": int(block["is_skip"].sum()),
            }
        )
    return pd.DataFrame(rows)


def load_coding(coding: Path) -> pd.DataFrame:
    """The two coders' independent scores, anonymised to coder_a / coder_b.

    The source files label their score columns differently and in opposite
    order, so they are selected by name rather than position.
    """
    a = pd.concat(
        [pd.read_excel(coding / f"coder_a_{i}.xlsx") for i in range(4)],
        ignore_index=True,
    )
    b = pd.concat(
        [pd.read_excel(coding / f"coder_b_{i}.xlsx") for i in range(4)],
        ignore_index=True,
    )
    assert (a["answer"].map(squash) == b["answer"].map(squash)).all(), (
        "the two coders' files are not row-for-row aligned"
    )
    return pd.DataFrame(
        {
            "interviewer": a["interviewer"],
            "answer": a["answer"],
            "specificity_coder_a": a["specificity"].astype(int),
            "specificity_coder_b": b["Specificity"].astype(int),
            "relevance_coder_a": a["relevance"].astype(int),
            "relevance_coder_b": b["Relevance"].astype(int),
        }
    )


def join_coding(coded: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Attach conversation_id and zone to each coded pair, by answer text."""
    answers = df[df["role"] == "user"]
    exact = defaultdict(list)
    for _, row in answers.iterrows():
        exact[squash(row["content"])].append((row["conversation_id"], row["zone"]))

    folded = [(r["conversation_id"], r["zone"], r["folded"]) for _, r in answers.iterrows()]
    resolved = []
    for text in coded["answer"]:
        hits = exact.get(squash(text), [])
        if len({cid for cid, _ in hits}) == 1:
            resolved.append(hits[0])
            continue
        target = fold(text)
        score, best = 0.0, None
        for cid, zone, candidate in folded:
            ratio = difflib.SequenceMatcher(None, target, candidate).ratio()
            if ratio > score:
                score, best = ratio, (cid, zone)
        assert score >= FUZZY_MIN, f"coded pair could not be joined (best {score:.2f})"
        resolved.append(best)

    out = coded.copy()
    out["conversation_id"] = [cid for cid, _ in resolved]
    out["zone"] = [zone for _, zone in resolved]
    return out


def krippendorff_alpha_ordinal(units: np.ndarray) -> float:
    """Krippendorff's alpha with the ordinal difference function.

    `units` is an (n_units, n_coders) array of integer ranks; no missing data.
    Follows Krippendorff (2012), ch. 12: alpha = 1 - D_observed / D_expected,
    where the ordinal metric weights a disagreement by the number of ranks
    spanned rather than treating all disagreements alike.
    """
    units = np.asarray(units, dtype=int)
    values = np.unique(units)
    index = {v: i for i, v in enumerate(values)}
    n_values = len(values)

    marginal = np.zeros(n_values)
    for value in units.ravel():
        marginal[index[value]] += 1
    n_total = marginal.sum()

    delta = np.zeros((n_values, n_values))
    for c in range(n_values):
        for k in range(c, n_values):
            span = marginal[c : k + 1].sum() - (marginal[c] + marginal[k]) / 2
            delta[c, k] = delta[k, c] = span**2

    d_observed = sum(
        sum(delta[index[x], index[y]] for x in row for y in row) / (len(row) - 1)
        for row in units
    ) / n_total
    d_expected = sum(
        marginal[c] * marginal[k] * delta[c, k]
        for c in range(n_values)
        for k in range(n_values)
    ) / (n_total * (n_total - 1))
    return 1 - d_observed / d_expected


def check_sources(messages: Path | None, coding: Path | None) -> None:
    """Fail with an explanation rather than a traceback from deep in pandas.

    The two inputs are deliberately absent from the public repository, so a
    reader who clones it and runs this script needs to be told that, not to
    debug a missing file.
    """
    def reachable(path: Path) -> bool:
        # One of these lives on a network share, where exists() itself raises
        # once the mount goes away rather than returning False.
        try:
            return path.exists()
        except OSError:
            return False

    missing = [
        f"  {name}: {'not given' if p is None else f'not found at {p}'}\n"
        for name, p in (("--messages", messages), ("--coding", coding))
        if p is None or not reachable(p)
    ]
    if not missing:
        return
    raise SystemExit(
        "build_data.py needs the private source data, which is not part of\n"
        "this repository:\n\n"
        + "".join(missing)
        + "\nPass --messages (the transcript export) and --coding (the folder "
        "holding\ncoder_a_{0..3}.xlsx and coder_b_{0..3}.xlsx), or set "
        "AINTERVIEWER_MESSAGES\nand AINTERVIEWER_CODING.\n"
        + "\nThe transcripts carry verbatim responses and the coding files "
        "carry the\ncoders' names, so neither can be published. The CSVs they "
        "produce are\nalready in the repository and the analysis in "
        "analysis.ipynb runs from\nthose alone -- this script is here to "
        "record how they were derived.\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for flag, env, help_ in [
        ("--messages", "AINTERVIEWER_MESSAGES", "message-level transcript export (.xlsx)"),
        ("--coding", "AINTERVIEWER_CODING", "folder with coder_a_{0..3}.xlsx and coder_b_{0..3}.xlsx"),
    ]:
        default = os.environ.get(env)
        parser.add_argument(
            flag,
            type=Path,
            default=Path(default) if default else None,
            help=f"{help_}; defaults to ${env}",
        )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_sources(args.messages, args.coding)
    df = load_messages(args.messages)
    coded = join_coding(load_coding(args.coding), df)

    kept = coded[coded["zone"] != "third_case"].reset_index(drop=True)
    kept.insert(0, "pair_id", range(len(kept)))
    for measure in ("specificity", "relevance"):
        kept[measure] = (
            kept[f"{measure}_coder_a"] + kept[f"{measure}_coder_b"]
        ) / 2

    agg = aggregate(df)
    agg.drop(columns="skip_clicks").to_csv(DATA / "agg.csv", index=False)

    kept[
        [
            "pair_id",
            "interviewer",
            "specificity_coder_a",
            "specificity_coder_b",
            "relevance_coder_a",
            "relevance_coder_b",
        ]
    ].to_csv(DATA / "coder_agreement.csv", index=False)

    kept[
        [
            "pair_id",
            "conversation_id",
            "interviewer",
            "specificity",
            "relevance",
        ]
    ].to_csv(DATA / "pair_level_scores.csv", index=False)

    per_interview = (
        kept.groupby("conversation_id")[["specificity", "relevance"]]
        .mean()
        .round(3)
        .reset_index()
        .merge(agg[["conversation_id", "Interviewer"]], on="conversation_id")
        .rename(columns={"Interviewer": "interviewer"})
    )
    per_interview[
        ["conversation_id", "interviewer", "relevance", "specificity"]
    ].to_csv(DATA / "relevance_and_specificity_scores.csv", index=False)

    print(f"interviews                {len(agg)}")
    print(f"analysed answers          {agg['question_count'].sum()}")
    print(agg.groupby("Interviewer")["question_count"].sum().to_string())
    print(f"skip-button clicks        {agg['skip_clicks'].sum()}")
    print(f"coded pairs               {len(kept)}")
    print(kept.groupby("interviewer").size().to_string())
    print(f"interviews with scores    {per_interview['conversation_id'].nunique()}")
    for measure in ("specificity", "relevance"):
        a = kept[f"{measure}_coder_a"].values
        b = kept[f"{measure}_coder_b"].values
        alpha = krippendorff_alpha_ordinal(np.column_stack([a, b]))
        print(
            f"{measure:12} exact {(a == b).mean():.3f}"
            f"  within 1 {(abs(a - b) <= 1).mean():.3f}  alpha {alpha:.3f}"
        )


if __name__ == "__main__":
    main()
