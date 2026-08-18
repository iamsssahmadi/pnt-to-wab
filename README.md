# PNT → WAB Subscore Predictor

A static, client-side web app that predicts 13 Western Aphasia Battery (WAB)
subscores from Philadelphia Naming Test (PNT) error-type counts alone (no
clinical covariates) — matching the model reported in the manuscript.

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

Per the manuscript, a simple linear model using only
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
## Disclaimer

For research and clinical decision-support purposes only. Not a substitute
for a full WAB administration or professional clinical judgment.
