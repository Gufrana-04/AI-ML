# PlaceMux — AI/ML Engineer Track (Week 2, Phase 2)

This repo covers **Task 1: Company Onboarding & Marketplace Data Model** and **Task 2: Job Posting with Skill Thresholds** for the PlaceMux student-job matching marketplace.

## What's in this repo

| File | Purpose |
|---|---|
| `students.csv` | 20 sample students with verified skill scores (0-100 scale) |
| `jobs.csv` | 9 sample job postings with required skill thresholds |
| `matches.csv` | All 180 student-job pairs with computed features and ground-truth labels |
| `placemux_task1_and_2.ipynb` | Full notebook: feature engineering, baseline, model, vectors, threshold validation, API contract |

## Dataset

- **students.csv**: 20 students across 10+ skill domains (Python/Data, Java/Backend, Frontend, Cloud, Security, Mobile, etc.)
- **jobs.csv**: 9 jobs, each requiring 3 skills with minimum threshold scores (L1-L100 scale) and a minimum experience requirement
- **matches.csv**: full cross-product of 20 students × 9 jobs = **180 pairs**, not a hand-picked sample

### How matches.csv is built

For every student-job pair, we compute:
- `skill_overlap_count` — how many required skills the student meets at or above the required score
- `skill_overlap_ratio` — overlap_count divided by total required skills (0.0–1.0)
- `experience_gap` — student's internship months converted to years, minus the job's minimum required years (positive = student exceeds requirement)

### Labeling rule (ground truth for `label`)

A pair is marked `label = 1` (good match) only if **both** are true:
1. `skill_overlap_ratio >= 0.6` (student meets at least 60% of required skills)
2. `experience_gap >= -0.5` (student is not more than half a year short on experience)

Otherwise `label = 0`. This rule is deterministic and documented so results are reproducible — it is *not* manually guessed per row.

**Resulting distribution:** 22 positive matches / 158 negative matches out of 180 pairs. This reflects realistic marketplace conditions — most students are not a fit for most jobs, since skills are domain-specific (a Frontend developer is correctly rejected for a Cloud Engineer role).

## Task 1 — Feature Space & API Contract

- **Baseline**: predict match if `skill_overlap_ratio >= 0.5` (no model, just a threshold)
- **Model**: Logistic Regression on `skill_overlap_ratio`, `experience_gap`, `education_level_encoded`
- **Evaluation**: held-out 80/20 train/test split, reporting precision, recall, false-positive rate
- **Explainability**: one-example walkthrough showing skill-by-skill pass/fail and plain-English reasoning
- **API contract**: documented request/response schema for `POST /api/v1/match`, handed off to the Backend team

## Task 2 — Match Vectors & Threshold Validation

- **Master skill list**: auto-extracted from all unique skills across students.csv and jobs.csv (no manual entry)
- **Vectorization**: each student/job converted into a fixed-length numeric vector following the master skill order; missing skill = 0
- **Similarity**: cosine similarity between student vector and job vector
- **Threshold validation**: per-skill pass/fail check (student score >= required score), plus an overall pass/fail
- **Competency mapping**: raw L1-L100 scores mapped to bands (*Novice* / *Beginner* / *Intermediate* / *Advanced* / *Expert*) for non-technical stakeholders
- **Validation**: threshold-based predictions checked against the 180-pair ground truth, with precision/recall/FPR reported

## How to run

1. Open `placemux_task1_and_2.ipynb` in Jupyter
2. Ensure `students.csv`, `jobs.csv`, and `matches.csv` are in the same folder as the notebook
3. Run all cells top to bottom

## Known limitations

- Sample size (180 pairs, 20 students, 9 jobs) is realistic for a demo but small for production-scale generalization claims
- Labeling rule is a single deterministic heuristic, not human-annotated ground truth — in production this would come from actual hiring outcomes or recruiter feedback
- No live FastAPI server is deployed; the API contract is specified but not yet implemented as a running endpoint
