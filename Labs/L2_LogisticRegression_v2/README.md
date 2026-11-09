# Lab 2 v2: logistic regression in 150 minutes

Start with **[L2_LogisticRegression_v2.qmd](L2_LogisticRegression_v2.qmd)** and
**[lab2_starter.m](lab2_starter.m)**. The original Lab 2 is unchanged.

## Student setup

- MATLAB R2019a or newer with Statistics and Machine Learning Toolbox.
- Keep `data/` with the saved script; run sections in order.
- Complete four expressions, TODO A–D, then write a three-line manual prediction
  and a six-line fit–predict–evaluate sequence for a new dataset. Loading, splitting,
  plotting, and output are supplied. Unfinished tasks deliberately stop the script.
- No Python, data download, or additional MATLAB add-ons. The dataset-reading
  task uses a linked webpage; all calculations use bundled data.
- Submit one screenshot-based PDF with the task deliverables and three short
  answers, as described in the worksheet. No separate theory write-up.

## Teaching outline

The short theory introduces only what the practical work needs: a linear score,
sigmoid probability, thresholded class, cross-entropy training, and held-out
accuracy/confusion counts. It continues Lab 1 v2's model → parameters → cost →
fitting → evaluation sequence, using `fitglm` as the only fitting interface.
The later tasks transfer the same ideas to a manual prediction and a second
dataset (Spine). A squared-feature comparison is optional. No manual gradient
descent, GUI workflow, or statistical inference is required.

| Activity | Minutes |
|---|---:|
| Setup and concise theory | 15 |
| Task 1: synthetic boundary and probabilities | 20 |
| Task 2: Titanic split and majority baseline | 15 |
| Task 3: one-predictor model, TODO A–C | 20 |
| Task 4: three-predictor comparison, TODO D | 20 |
| Task 5: calculate a prediction | 10 |
| Task 6: dataset reading and independent Spine workflow | 25 |
| Task 7: self-check and screenshot PDF assembly | 15 |
| Troubleshooting/buffer throughout | 10 |
| **Total** | **150** |

If support takes longer, supply an individual TODO expression rather than add
homework. The squared-feature extension and threshold discussion are optional; prioritize
the six-line transfer task and in-session report assembly.

Both Titanic models use the same 100 test passengers and threshold 0.5.
The independent Spine task uses a new model, all 12 numeric predictors, and a fixed
248/62 stratified split. Do not interpret it as a medical diagnostic tool.
The prepared bundle removes the original split's duplicate passenger from
training (791 remaining). This fixes the known row overlap, not the unknown
upstream preprocessing. Source coding, preprocessing and licensing limits are
explicit in `data/README.md`; confirm source terms before broader redistribution.
Treat the results as a teaching comparison, not a validated survival predictor.
No answer key or completed student solution is bundled.

## Files and maintenance

- `L2_LogisticRegression_v2.qmd`: editable worksheet with non-executing examples.
- `.html` / `.pdf`: reading copies; HTML embeds its resources and uses MathML.
- `lab2_starter.m`: MATLAB execution surface with four TODOs and two short independent code blocks.
- `data/`: prepared CSVs, original Titanic snapshots, and provenance notes.
- `prepare_data.py`: offline instructor-only regeneration, standard library only.
- `test_data.py`: offline integrity/reproducibility tests for all three datasets.

```sh
python -B -m unittest discover -s /path/to/L2_LogisticRegression_v2 -p test_data.py -v
quarto render L2_LogisticRegression_v2.qmd --to html
quarto render L2_LogisticRegression_v2.qmd --to pdf
```

PDF rendering needs a configured TeX installation. Rendering does not execute
or validate the MATLAB script. Students run the saved `.m` file, not a notebook.

## Validation

All eleven integrity/reproducibility tests pass, including Spine split, source
mapping, invalid-input, and regeneration checks. An independent academic review
checked theory, workload, task alignment, and source-level MATLAB usage.

A temporary completed copy of the starter ran through all sections in Octave
9.4 using temporary `readmatrix`, binomial `fitglm`, and `predict` compatibility
functions. Both figures exported successfully; probability bounds, dimensions,
training-cost order, and confusion counts passed checks. Independent NumPy
Newton fits agreed with the results. The added manual prediction, six-line Spine workflow, and optional squared-feature
model also passed a fresh compatibility run. The latest all-12-predictor Spine
run used temporary scaled, pseudoinverse-based IRLS to handle near-redundant
columns; this is not MATLAB's solver and its exact metrics are not a native
MATLAB reference. See `data/README.md` for the conditioning caveat.
An unfinished copy stopped at the intended TODO A assertion. Neither completed code nor compatibility functions is shipped.
This validates calculations and control flow, **not native MATLAB/toolbox
execution**, which remains a validation gap.

HTML and PDF rendered successfully; theory and task-table pages
received a visual layout spot-check. Rendering does not run the starter.
