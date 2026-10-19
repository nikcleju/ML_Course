# Lab 1 v2: linear regression in 150 minutes

Start with **[L1_LinearRegression_v2.qmd](L1_LinearRegression_v2.qmd)** and
**[lab1_starter.m](lab1_starter.m)**. All data are bundled in `data/`.

## Student setup

- MATLAB R2019a or newer with Statistics and Machine Learning Toolbox.
- Open the saved starter script and run its sections in order.
- Four TODO expressions intentionally stop execution until completed; this is a
  worksheet, not a ready-to-run solution. Everything else is supplied.
- No Python, internet connection, extra MATLAB add-ons, or data downloads.
- Each task lists its deliverables. Compile the listed screenshots and three
  short self-check answers into one PDF (`Lab1_Name_Group.pdf`) and send it to
  the instructor. Capture Task 2's scatterplot before Task 4 modifies it.
  No theory write-up or separate source-code submission is required unless requested.

## Instructor preparation

Check toolbox availability before class. Demonstrate Run Section and indexing
before starting the first task. Budget 20 of the 150 minutes for support.
Use the final 15-minute block for the short self-check and
screenshot-report assembly, rather than adding homework or new coding tasks.
If students need more help, supply a TODO expression individually rather than
adding mandatory work outside the session. No separate answer key is bundled.

The worksheet follows **model → parameters → training cost → cost minimization
→ held-out evaluation**. Tasks use explicit goals, steps with practical checks, and deliverables.
The synthetic example compares an untrained candidate line with the fitted line
and prints both MSE costs; the car comparison distinguishes training MSE from
test RMSE. This adds no extra TODOs or session time.

The core deliberately uses one fitting method (`fitlm`), a visual synthetic
introduction, a fixed holdout, a training-mean baseline, and only two prescribed
real-data models. Students identify the bias and weights, understand the training
cost, and evaluate the fitted models on the held-out rows.

The optional item is discussion only. Model-comparison conclusions must describe
this split, not imply that more predictors always improve unseen prediction.
The lab does not cover inference statistics or causality.

## Files and maintenance

- `L1_LinearRegression_v2.qmd`: editable student worksheet; non-executing MATLAB examples.
- `L1_LinearRegression_v2.html` / `.pdf`: rendered reading copies. The HTML
  uses embedded resources and native MathML, with no external math-script dependency.
- `lab1_starter.m`: guided MATLAB script with TODO A–D.
- `data/auto-mpg.data`: complete 398-record source snapshot.
- `data/auto_mpg_lab.csv`: prepared numeric data, fixed split and source row IDs.
- `data/synthetic_line.csv`: fixed 40-observation introductory example.
- `data/README.md`: attribution, licence, units, omissions, and regeneration details.
- `prepare_data.py`: offline data preparation; instructor use only.
- `test_data.py`: offline integrity and reproducibility tests; instructor use only.

Run maintenance checks with `python -B -m unittest discover -s /path/to/this/folder -p test_data.py`.
Render the worksheet from this folder using `quarto render L1_LinearRegression_v2.qmd --to html`
or `--to pdf` (PDF requires a configured TeX installation). Rendering does not execute
or validate the MATLAB script. An executable notebook is intentionally not provided:
the saved `.m` script is the execution surface.

## Validation

The five integrity/reproducibility tests pass. Reference least-squares results
were independently cross-checked with NumPy and Octave. A temporary completed
copy of the starter ran through all sections in Octave using temporary
`readmatrix`, `fitlm`, and `predict` compatibility functions and headless plots.
Those compatibility functions are not shipped or required by students. This
checks the script's dimensions, calculations, and flow, but is **not** a native
MATLAB/toolbox execution test. Both reading formats rendered successfully;
the PDF received a spot layout check. Native MATLAB remains the validation gap.
