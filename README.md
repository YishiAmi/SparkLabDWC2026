# SparkLab – DataWorks Challenge 2026
**Machine-learning-assisted history matching and production forecasting** (ExxonMobil DWC26)

## What this does
1. **Proxy model (machine learning)** – `src/proxy.py`. PCA + Gaussian-process regression maps the 4 reservoir uncertainty settings
   (fault transmissibility, porosity multiplier, permeability multiplier, aquifer pore volume) to the full 1998-2008 gas/oil/water
   production curves (cumulative and rate) in milliseconds instead of a simulator run.
2. **History matching** – `src/hm.py`. 100,000 candidate settings are scored by the proxy against the observed field history, best ones polished with Nelder-Mead.
3. **Forecast** – `src/decline.py` + `src/forecast_run.py`. The 10 best, diverse parameter sets are extrapolated from 2008 to 01/01/2028
   (oil: mean of exponential & hyperbolic decline; gas: gas-oil-ratio trend; water: ln(WOR) vs cumulative oil) and written into the official template.

## Data is NOT included
The datasets and starter kit are provided exclusively for the DataWorks Challenge 2026 and must not be published, so this repository contains **code only**.
To reproduce, download the starter kit from the challenge web app and unzip it so that you have a folder containing `TestCases/`.

## Folder layout
| Folder | Contents |
|---|---|
| `src/` | all code (see run order) |
| `data/` | (empty) created files go here: `prod.pkl`, `unc.pkl`, `data.npz`; **you copy the official `09 Template Deliverable.xlsx` here** |
| `models/` | (empty) trained proxy model is written here |
| `outputs/` | (empty) filled template, Power BI data, model comparison, plots are written here |
| `docs/` | `model_comparison.md` |

## How to run (Python 3.10+, `pip install -r requirements.txt`)
```
export DWC_RAW=/path/to/starter_kit           # folder that contains TestCases/   (Windows: set DWC_RAW=...)
cp "$DWC_RAW/TestCases/09 Template Deliverable.xlsx" data/
python src/00_extract.py      # TestCases/01, 02 -> data/prod.pkl, unc.pkl
python src/prep.py            # -> data/data.npz
python src/proxy.py           # optional: validation report of the proxy (~1 min)
python src/hm.py              # history matching -> models/proxy_final.pkl (~15 s)
python src/forecast_run.py    # 10 best cases -> 80-quarter forecast -> outputs/09 Template Deliverable - Filled.xlsx
python src/compare_models.py  # optional: GP vs ridge / MLP / random forest / gradient boosting
python src/build_powerbi.py   # outputs/PowerBI_Data.xlsx
python src/make_plot.py       # outputs/forecast_overview.png
```
Optional grid analysis (grid files 03-08):
```
python src/extract_grid.py    # -> data/grid/case_NNN.npz   (~10 min, ~1 GB)
python src/grid_summary.py    # -> outputs/grid_summary.csv
python src/grid_vary.py ; python src/grid_vary2.py ; python src/grid_vary3.py
```

## Results
* **Proxy accuracy on unseen cases** (trained on the 70 train cases; mean absolute % error): validation 15 cases / test 15 cases –
  oil & gas cumulative ≈ 0.12 %, water cumulative 0.37-0.56 %. Compared with ridge/polynomial (1-8 %), random forest, gradient boosting (0.4-2 %) and a neural network (0.2-0.75 %): see `docs/model_comparison.md`.
* **History match**: porosity multiplier is well constrained (≈ 1.33); permeability multiplier, aquifer volume and fault transmissibility are only weakly constrained by the history.
* **Forecast (mean of the 10 cases)**: oil 38.4 M STB (2008) → ≈ 51.3 M (2018) → ≈ 57.7 M (2028); gas 13.9 M → 18.6 M → 21.0 M MSCF; water 12.8 M → 43.2 M → 79.5 M STB.

## Findings about the data
* The only information in the grid files (03-08) is the 4 settings applied to a fixed grid: 24 of 29 columns are identical in all 100 cases; PORO/PORV scale with the porosity multiplier (PORV also with aquifer volume in 2,286 aquifer-edge cells), PERMX/PERMY/TRANX/TRANY scale with the permeability multiplier in one zone of 11,575 cells. Fault transmissibility does not appear in the grid. So the grid cannot add information beyond the 4 settings.
* The simulated production in the files ends at 01/01/2008; there is no ground truth for 2008-2028.

## Limitations (please read)
* **The forecast beyond 2008 is an extrapolation**, not a learned prediction – no post-2008 training labels exist. The decline-curve method was only back-tested at 1-3 year horizons (cut history at 2005-2007 and forecast to 2008: oil error ≈ 0.2-1 %). Uncertainty grows with horizon, especially for water and after 2018.
* The 10 submitted cases agree within ≈ 1.5 % because they share one extrapolation method; this shows consistency, not correctness.
* First 6 quarters of the rate curves are excluded from the history-match score (simulated production starts one quarter later than the observed history).
* Quarterly interpolation of the simulator's irregular timesteps; water-injection rate in the history sheet was not used.
