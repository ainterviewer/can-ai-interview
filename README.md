# Are AI-led interviews comparable to human-led interviews?

This repository contains the data, code, and documentation for a study comparing AI-led and human-led semi-structured interviews on the topic of university students' academic help-seeking behavior. N=41 university students were randomly assigned to either a human-led or AI-led interview, both conducted via a text-based chat interface. The AI interview system used OpenHermes 2.5 (a fine-tuned version of Mistral 7B) hosted on GDPR-compliant servers, following the same interview guide as the human interviewers.

## Repository Contents

Data files live in `data/`, the code that produces and checks them in
`scripts/`, and the manuscript in `paper/`. `analysis.ipynb` stays at the top
level as the entry point; it reads from `data/` and writes `figures/plots.png`.

| File / Folder                               | Description                                                                |
| ------------------------------------------- | -------------------------------------------------------------------------- |
| `analysis.ipynb`                            | Jupyter notebook containing all analysis code                              |
| `requirements.txt`                          | Python dependencies                                                        |
| `data/agg.csv`                              | Aggregated behavioral metrics per interview (41 conversations)             |
| `data/relevance_and_specificity_scores.csv` | Aggregated coded quality scores per interview                              |
| `data/coder_agreement.csv`                  | Both coders' independent scores for every coded pair                       |
| `data/pair_level_scores.csv`                | Coded scores with the interview each answer belongs to                     |
| `data/interview_day.csv`                    | Day of data collection per interview, as an index with no calendar date    |
| `data/interview_guide.json`                 | Full interview guide including framing, questions, probes, and intro/outro |
| `prompts/`                                  | Jinja prompt templates used by the AI interview system                     |
| `figures/`                                  | Figures used by the manuscript; `plots.png` is written by the notebook     |
| `paper/`                                    | LaTeX source and bibliography for the manuscript                           |
| `scripts/build_data.py`                     | Rebuilds the CSVs above from the private source data (see Reproducing)     |
| `scripts/stats_tables.py`                   | Prints the condition comparison, the equivalence bounds and the day-structure robustness checks |
| `scripts/verify_paper.py`                   | Checks every number printed in the manuscript against the data             |

## Data

### data/agg.csv

Aggregated behavioral metrics per conversation. Raw interview transcripts are not shared to protect participant confidentiality (see Data Availability).

| Column                                    | Description                                                   |
| ----------------------------------------- | ------------------------------------------------------------- |
| conversation_id                           | A unique ID per conversation                                  |
| Interviewer                               | If the interviewer was either `Human` or `AI`                 |
| duration(min)                             | Minutes of the analysed portion of the interview              |
| question_count                            | Number of questions answered                                  |
| total_character_by_user                   | Total characters in the respondent's answers                  |
| avg_response_length                       | Mean characters per answer                                    |
| questions_per_min                         | Questions answered per minute of interview                    |
| characters_per_min                        | Characters produced by the respondent per minute of interview |
| avg_response_duration_by_interviewer(sec) | Mean seconds the interviewer took to send the next question   |
| avg_response_duration_by_user(sec)        | Mean seconds the respondent took to send an answer            |

### data/relevance_and_specificity_scores.csv

Aggregated coded quality scores per conversation. Scores represent averages across all answers coded within an interview.

| Column          | Description                                                                             |
| --------------- | --------------------------------------------------------------------------------------- |
| conversation_id | A unique ID per conversation                                                            |
| interviewer     | If the interviewer was either `Human` or `AI`                                           |
| relevance       | Average relevance score for the interview (1–5; 1 = very irrelevant, 5 = very relevant) |
| specificity     | Average specificity score for the interview (1–5; 1 = very vague, 5 = very specific)    |

### data/coder_agreement.csv

Every coded question-answer pair was scored independently by two coders; the score used in the analysis is the mean of the two. This file holds both sets of scores so that inter-rater reliability can be recomputed. It contains no response text.

| Column              | Description                                   |
| ------------------- | --------------------------------------------- |
| pair_id             | Identifier for the coded question-answer pair |
| interviewer         | If the interviewer was either `Human` or `AI` |
| specificity_coder_a | First coder's specificity score (1–5)         |
| specificity_coder_b | Second coder's specificity score (1–5)        |
| relevance_coder_a   | First coder's relevance score (1–5)           |
| relevance_coder_b   | Second coder's relevance score (1–5)          |

### data/pair_level_scores.csv

Coded scores with the interview each answer belongs to. Coded answers are matched back to the corpus on normalised text, falling back to closest-match where an answer recurs verbatim; `build_data.py` requires every coded pair to resolve rather than dropping those it cannot place. It contains no response text.

| Column          | Description                                               |
| --------------- | --------------------------------------------------------- |
| pair_id         | Identifier shared with `coder_agreement.csv`              |
| conversation_id | A unique ID per conversation                              |
| interviewer     | If the interviewer was either `Human` or `AI`             |
| specificity     | Mean of the two coders' specificity scores (1–5)          |
| relevance       | Mean of the two coders' relevance scores (1–5)            |

Neither a guide-section index nor a main-question/probe flag is published. Both distinctions are recorded by the platform for AI-led interviews but were reconstructed by hand for human-led ones, so they are not measured the same way in the two conditions and should not be used to compare them. The analysis does not rely on them: the one exclusion it applies is located by matching the text of a scripted question, which is recorded identically in both conditions.

## Scope of the data

A third substantive case, in which the student asked for help but did not receive it, was administered during the first three days of data collection and then dropped from both conditions. The interviews were scheduled in half-hour slots and the human interviewers rarely reached it, so keeping it would have meant comparing the conditions on material only one of them regularly covered. It reached eight of the 41 interviews, and every file here excludes it; the duration and rate columns in `agg.csv` are computed over the same span, so the rates put analysed answers over the time in which they were collected.

The closing prompt is kept. It was administered in 18 of 20 AI-led and 20 of 21 human-led interviews, so there is no comparability argument for removing it.

Answers consisting of the `<|skipquestion|>` control token are excluded from the answer counts. These are clicks on the interface's skip button rather than responses; there are 22, all in AI-led interviews.

### data/interview_day.csv

Randomisation was performed separately for each day of data collection, so the day is a design variable. This file maps each interview to a day index derived from message timestamps. The index deliberately carries no calendar date.

| Column          | Description                                        |
| --------------- | -------------------------------------------------- |
| conversation_id | A unique ID per conversation                       |
| interview_day   | Day of data collection, indexed 1–10 in date order |

Respondent demographics (age, gender, region of origin, study programme, degree level, prior ChatGPT use) are **not** included in this repository. In a sample of 41 that combination is close to uniquely identifying, and it could be joined to interview behaviour and quality scores on `conversation_id`. They are reported in aggregate in the paper and are available from the authors under a data-sharing agreement.

## Analysis Code

All analysis code can be found in [analysis.ipynb](./analysis.ipynb). Dependencies can be found in `requirements.txt`.

### Reproducing the analysis

The notebook runs from the CSVs in this repository and needs nothing else:

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt nbconvert
jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
```

#### With uv

[uv](https://docs.astral.sh/uv/) picks up Python 3.12 from `.python-version`
and downloads it if it is not installed:

```sh
uv venv
uv pip install -r requirements.txt
uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
uv run python scripts/verify_paper.py
```

`nbconvert` is only needed to run the notebook from the command line, so it is
not pinned in `requirements.txt`. To work in the notebook interactively, point
your editor at `.venv` or run `uv run --with jupyterlab jupyter lab`.

It reproduces every number the manuscript reports and rewrites
`figures/plots.png`. [stats_tables.py](./scripts/stats_tables.py) prints the condition
comparison and the equivalence bounds on their own, without the notebook, together
with the robustness checks of Appendix C: the comparison repeated within days
(van Elteren) and without the first day of data collection, on which only the
human condition was run.

[build_data.py](./scripts/build_data.py) documents how the CSVs were derived, but
**cannot be run from this repository**. It reads the message-level transcripts
and the two coders' score sheets, and neither can be published: the transcripts
carry verbatim responses given under a confidentiality agreement, and the
coding files identify the coders by name. The script is included because the
derivation was previously undocumented, which is how eleven coded pairs came
to be lost to a silent join failure. With access to the private sources, pass
their locations explicitly (or set `AINTERVIEWER_MESSAGES` and
`AINTERVIEWER_CODING`); the script has no default paths:

```sh
python scripts/build_data.py \
  --messages path/to/filtered_messages.xlsx \
  --coding path/to/coding   # holds coder_a_{0..3}.xlsx and coder_b_{0..3}.xlsx
```

Run against the private sources it reports:

```
analysed answers  909   (AI 618, Human 291)
coded pairs       395   (AI 200, Human 195)
skip-button clicks 22
specificity  exact 0.527  within 1 0.937  alpha 0.648
relevance    exact 0.934  within 1 0.980  alpha 0.365
```

### Checking the manuscript against the data

Every figure the manuscript reports is derived from the CSVs here, so the two
can drift apart whenever the analysis is re-run, and at various points they
have. [verify_paper.py](./scripts/verify_paper.py) recomputes each reported quantity,
formats it the way the manuscript does, and checks it against `paper/main.tex`:

```sh
python scripts/verify_paper.py   # non-zero exit if anything has drifted
```

It covers the sample and corpus sizes, every cell of the condition-comparison
and inter-rater tables, the descriptive values quoted in the Results, and the
regression coefficients and their standard errors. Table values are matched
by row rather than by searching the file, because a bare `0.008` also occurs
as an adjusted R-squared, and a check that can pass on the wrong cell is not
a check.

It guards against stale numbers rather than testing the analysis: a pass means
the paper agrees with the data, not that either is correct.

Multiplicity is controlled with Holm's step-down procedure across the nine
comparisons. Holm holds the family-wise error rate at the same level as
Bonferroni under no additional assumptions and is uniformly more powerful;
the notebook reports both, and no conclusion differs between them.

## Paper

The manuscript source is in `paper/`. Build it with:

```sh
cd paper
latexmk -pdf main.tex
```

`latexmk` runs `pdflatex` and `bibtex` as many times as the cross-references and citations need, which a single `pdflatex` run does not. Building without it produces a PDF whose citations render as `(?)` and whose bibliography is missing entirely. Use `latexmk -c` to clear the build artefacts.

The build needs a TeX distribution with `latexmk`, `natbib`, `hyperref`, `tabularx`, `booktabs`, `caption`, `float`, `enumitem`, `geometry`, `setspace` and the `times` (PSNFSS) fonts — all present in a full TeX Live install. On Arch, `latexmk` ships in `texlive-binextra`. The bibliography style `asa.bst` is included in `paper/` because it is not part of TeX Live.

Figures live in the top-level `figures/` directory, not inside `paper/`, so that there is a single copy: `plots.png` is written there by the final cell of [analysis.ipynb](./analysis.ipynb), and `main.tex` reads it back out via `../figures/`. `figure_1.png` and `interview_flow.pdf` are static.

## AI Interview System — Prompts

The AI interviewer was implemented using three LLM agents working in tandem. Prompt templates are stored in the `prompts/` directory as Jinja (`.jinja`) files with named variable slots.

### Probing Agent

The probing agent is responsible for asking interview questions and follow-up probes. It receives the current question context, suggested probes from the interview guide, and the conversation transcript so far, and generates a single follow-up question.

**System prompt** (`prompts/probing_agent_system_prompt.jinja`):

> Establishes the agent as an experienced qualitative social science interviewer whose sole task is to ask relevant follow-up probes one question at a time.

**Probing prompt** (`prompts/probing_agent_probing_prompt.jinja`):

Template variables:

- `question_context` — the current main question and its associated metadata from the interview guide
- `interview_transcript` — the conversation transcript up to the current point
- `probes` — the list of suggested follow-up probes from the interview guide

Behavioral constraints encoded in the prompt:

- Ask only one question at a time; keep questions brief
- Ask open-ended questions; avoid yes/no questions
- Do not give advice or feedback to the interviewee
- When selecting a probe from the suggestions, paraphrase it to fit the conversation naturally

### Reformulation Agent

It rephrases the main question to incorporate information already provided by the interviewee (`prompts/reformulation_prompt.jinja`).

### Classification Agent

The classification agent makes binary decisions that control interview flow — specifically, whether the interviewee's last response satisfies the condition required to advance to the next question (e.g., whether a yes/no question was answered affirmatively, triggering a section skip).

**System prompt** (`prompts/classification_agent_system_prompt.jinja`):

> Establishes the agent as a binary classifier that returns only `0` or `1`.

**Classification prompt** (`prompts/classification_agent_classification_prompt.jinja`):

Template variables:

- `conversation_history` — prior context of the interview
- `text` — the specific response to classify
- `next_question_instruction` — a natural-language description of the classification criterion
- `classification_examples` — optional few-shot examples (JSON)

The agent returns `1` if the text meets the criterion and `0` otherwise.

---

## Data Availability

All materials underlying this study are publicly available at:

**https://github.com/ainterviewer/can-ai-interview**

The repository includes:

- **Aggregated data**: `data/agg.csv` (behavioral metrics) and `data/relevance_and_specificity_scores.csv` (coded quality scores)
- **Interview instrument**: `data/interview_guide.json` — the full semi-structured interview guide including framing, introductory and closing texts, all main questions, follow-up probes, and conditional skip logic
- **AI system prompts**: Jinja prompt templates for the probing and classification agents (`prompts/`)
- **Analysis code**: `analysis.ipynb` with all statistical analyses and `requirements.txt`

**Materials not shared**: Raw interview transcripts are not shared to protect the confidentiality of participants. Interview responses were collected from university students under a confidentiality agreement, and sharing verbatim transcripts would risk re-identification.

---

## Interview Guide

The following is the full interview guide used in both the human-led and AI-led interviews, reproduced from `data/interview_guide.json`.

### Study Context

This interview investigates university students' academic help-seeking behavior. The guide has two main sections and a closing: (1) a situation where the student needed and received academic help, (2) a situation where the student needed help but did not ask for it, and (3) a brief closing for any additional reflections. The interview examines the social processes and mechanisms surrounding help-seeking: why students need help, who they turn to, the social situation, their relationship to helpers, and their feelings, attitudes, and behaviors. A secondary aim is to understand how students navigate their social networks to access resources and cope with academic challenges.

### Introduction

> Thank you very much for taking part in our research. In this interview, we will ask you open-ended questions to investigate university students' academic help seeking behavior. It is very common for students to face various academic challenges in their studies such as understanding a concept, solving an exercise, doing a group project and so on.
>
> There are no right or wrong answers. We are very interested in hearing your experience and perspective. The only constraint is that you must write your answers in one go, not using multiple lines. Also, your responses will remain confidential and all personally identifiable information will be removed from publications.

---

### Section 1 — Needed Help and Got Help

_Section aim_: To understand the social situation in which university students needed and were able to get academic help — including reasons for needing help, how students identify when help is needed, what social competencies they use to decide who to approach, and the social processes involved.

**Q1.** Can you think of a situation where you needed academic help and got help with your studies? _(yes/no; if no, section is skipped)_

**Q2.** What did you specifically need help with?

**Q3.** How did you get help?

- Did you personally ask for help? If so, how?
- From whom did you get help? Can you describe the social situation?
- How would you describe your relation to them?
- How do you decide who can help best and how to approach them?

**Q4.** Who else did you consider asking for help?

- What factors influenced your decision?

**Q5.** How did you feel after getting the help?

- What are your expectations from the helper?

**Q6.** In general, what are the reasons you seek academic help?

**Q7.** Who do you ask to get help to resolve your academic challenges? How would you describe your relation to them?

---

### Section 2 — Needed Help but Did Not Ask

_Section aim_: To understand situations where university students refrained from asking for academic help — including why students hold back, their relation to potential helpers, what factors affect their decision, and the social processes involved.

**Q1.** Can you think of a situation where you needed help with your studies but did not ask for it? _(yes/no; if no, section is skipped)_

**Q2.** What did you specifically need help with?

**Q3.** Although you did not ask for help, did you consider asking anyone for help? And if so, who?

- How would you describe your relation to them?
- What were your considerations around asking for help?

**Q4.** Why did you not prefer asking for help?

- When do you feel the most comfortable to ask for academic help?
- When do you feel the least comfortable to ask for academic help?

**Q5.** How did you resolve the issue?

- Did you end up getting any help? Can you describe the social situation?
- How do you decide who can help best and how to approach them?
- When do you feel the most comfortable to ask for academic help?
- When do you feel the least comfortable to ask for academic help?
- Why did you not prefer asking for help?

**Q6.** How did you feel about this situation?

- Under what conditions would you seek help? Can you explain why?
- What changes would you suggest at the university to better meet your academic needs?

**Q7.** In general, in which cases or from whom do you refrain from asking academic help? Why so?

---

### Section 3 — Closing

**Q1.** Before we conclude our interview, is there anything you'd like to add or any perspectives you think we might have missed? _(yes/no; if no, interview closes)_

---

### Closing

> Thank you very much for sharing your insights and experiences. Your input is a valuable contribution to our research.

---

### Interview Guide Notes

- Each main question may be followed by a number of AI-generated follow-up probes (up to the `max_probes_n` limit specified per question). For human-led interviews, the probes listed above served as guidance for the interviewer.
- Conditional skip logic: if the interviewee answers "no" to the opening screening question of Sections 1 or 2, the remaining questions in that section are skipped.
- The probes listed under each question are suggestions; both human interviewers and the AI probing agent were expected to paraphrase and adapt them to fit the natural flow of the conversation.
