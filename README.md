# PNT → WAB Subscore Predictor

A static, client-side web app that predicts 13 Western Aphasia Battery (WAB)
subscores from Philadelphia Naming Test (PNT) error-type counts alone (no
clinical covariates) — matching the "M1, PNT-only" model reported in the
accompanying manuscript.

**Status:** local-only. This is not currently deployed to GitHub Pages or
any other host; run it locally per "Running locally" below. See
"Deploying to GitHub Pages" if you choose to publish it later.

## Privacy

All inference runs **in the visitor's browser** via [onnxruntime-web](https://github.com/microsoft/onnxruntime)
(WebAssembly). The PNT counts you enter, and the predictions produced, never
leave your device — there is no backend, no API call, and no analytics.
The page works fully offline once loaded. This is why the models were
converted from their original Python format (scikit-learn / XGBoost /
LightGBM / CatBoost) into [ONNX](https://onnx.ai/), rather than the site
calling out to a Python server.

## How it works

- `index.html` / `style.css` — UI: PNT count inputs and a results table.
- `app.js` — reads inputs, reproduces the exact 29-feature engineering
  pipeline used at training time (error ratios, percentages, composites,
  interactions), then runs all 13 ONNX models locally via `onnxruntime-web`
  and renders the predicted subscores.
- `models/*.onnx` + `models/config.json` — the 13 trained models (one
  algorithm per WAB subscore, selected via the leakage-free nested
  cross-validation procedure described in the manuscript, §2.3.1–§2.3.2)
  converted to ONNX, plus their feature order / max scores / CV R².
- `lib/` — vendored `onnxruntime-web` runtime (JS + WASM), served locally
  so the page never depends on a third-party CDN at runtime.

Model → subscore mapping (final algorithm per target, corrected cross-validated R²):

| Subscore | Model | CV R² |
|---|---|---|
| WABNaming subscore | CatBoost | 0.899 |
| WABObject Naming | CatBoost | 0.886 |
| WABInformation Content | XGBoost | 0.769 |
| WABSpontaneous Speech Total | CatBoost | 0.766 |
| WABResponsive Speech | CatBoost | 0.756 |
| WABRepetition Subscore | CatBoost | 0.677 |
| WABSentence Completion | CatBoost | 0.672 |
| WABWord Fluency | RandomForest | 0.617 |
| WABFluency Rating | RandomForest | 0.583 |
| WABComprehension Subscore | ElasticNet | 0.499 |
| WABComprehension Auditory Words | ElasticNet | 0.465 |
| WABComprehension Sequential Commands | ElasticNet | 0.413 |
| WABComprehension Yes/No | ElasticNet | 0.240 |

Per the manuscript (§3.2, §4.5), a simple linear model using only
percent-correct naming matched or exceeded this ensemble for 7 of these 13
subscores — the ensemble's advantage is concentrated in the naming and
connected-speech outcomes at the top of this table.

## Running locally

No build step or dependencies are required — it's static HTML/JS.

```bash
cd site
python -m http.server 8000
# open http://localhost:8000
```

(Serving over `file://` directly will not work because browsers block
`fetch()` of local files under that scheme; use any static file server.)

## Deploying to GitHub Pages

1. Push this folder's contents to a GitHub repository (this folder *is*
   the repo root).
2. In the repo: **Settings → Pages → Build and deployment → Deploy from
   branch**, branch `main`, folder `/ (root)`.
3. Wait for the Pages build to finish; the site will be live at the URL
   GitHub shows on that settings page.

No GitHub Actions workflow, server, or backend is needed — everything is
static files.

## Regenerating the models

The `.onnx` files here were produced by converting the trained `.pkl`
models in the parent research repo's `models_m1/` (see `train_model1_final.py`,
which reads `model1_results/tables/final_model_config.json` — the
leakage-free nested-selection output of `run_model1_only.py` — and fits one
final model per target on the full dataset) with `convert_to_onnx.py
models_m1` (`skl2onnx` / `onnxmltools` for RandomForest, XGBoost, LightGBM,
ElasticNet; CatBoost's built-in ONNX exporter for CatBoost). If the
underlying models are retrained, re-run that pipeline and re-run the
conversion to replace the contents of `models/`.

An older `models/` directory (M2: PNT + Days-Post-Stroke + Aphasia Type
covariates) still exists in the parent repo as an archive from an earlier,
since-abandoned direction — it is unrelated to this site and to the
manuscript's reported results.

## Disclaimer

For research and clinical decision-support purposes only. Not a substitute
for a full WAB administration or professional clinical judgment.
