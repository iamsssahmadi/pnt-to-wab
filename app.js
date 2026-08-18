// PNT -> WAB predictor. Everything below runs entirely client-side:
// PNT counts and predictions never leave the browser.

ort.env.wasm.wasmPaths = "lib/";
ort.env.wasm.numThreads = 1; // avoids requiring cross-origin-isolation (SharedArrayBuffer) on GitHub Pages
ort.env.wasm.simd = true;

const PNT_FEATURES = [
  "pntCorrect", "pntFormal", "pntMixed", "pntNR",
  "pntNeologism", "pntSemantic", "pntUnrelated",
];

const PNT_LABELS = {
  pntCorrect: "Correct",
  pntFormal: "Formal Paraphasias",
  pntMixed: "Mixed Paraphasias",
  pntNR: "No Response",
  pntNeologism: "Neologisms",
  pntSemantic: "Semantic Paraphasias",
  pntUnrelated: "Unrelated Responses",
};

const EPS = 1e-8;

let config = null;
const sessions = {}; // target -> ort.InferenceSession
let lastPreds = null; // target -> number

function safeName(target) {
  return target.replace(/ /g, "_").replace(/\//g, "_");
}

// Mirrors build_m1_features() in train_model1_final.py exactly.
function engineerFeatureRow(pntVals) {
  const c = pntVals.pntCorrect, fo = pntVals.pntFormal, mi = pntVals.pntMixed,
        nr = pntVals.pntNR, ne = pntVals.pntNeologism, se = pntVals.pntSemantic,
        un = pntVals.pntUnrelated;

  const totalErr = fo + mi + nr + ne + se + un;
  const totalResp = c + fo + mi + nr + ne + se + un;

  return {
    pntCorrect: c, pntFormal: fo, pntMixed: mi, pntNR: nr,
    pntNeologism: ne, pntSemantic: se, pntUnrelated: un,

    pntFormal_ratio_err: fo / (totalErr + EPS),
    pntMixed_ratio_err: mi / (totalErr + EPS),
    pntNR_ratio_err: nr / (totalErr + EPS),
    pntNeologism_ratio_err: ne / (totalErr + EPS),
    pntSemantic_ratio_err: se / (totalErr + EPS),
    pntUnrelated_ratio_err: un / (totalErr + EPS),

    pntCorrect_pct: c / (totalResp + EPS),
    pntFormal_pct: fo / (totalResp + EPS),
    pntMixed_pct: mi / (totalResp + EPS),
    pntNR_pct: nr / (totalResp + EPS),
    pntNeologism_pct: ne / (totalResp + EPS),
    pntSemantic_pct: se / (totalResp + EPS),
    pntUnrelated_pct: un / (totalResp + EPS),

    correct_pct: c / (totalResp + EPS),
    error_pct: totalErr / (totalResp + EPS),
    correct_err_ratio: c / (totalErr + EPS),

    semantic_composite: se + un,
    phonological_composite: fo + ne,
    mixed_composite: mi,
    other_composite: nr,

    correct_x_formal: c * fo,
    semantic_x_phonological: (se + un) * (fo + ne),
  };
}

function buildFeatureArray(row, featureNames) {
  return Float32Array.from(featureNames.map((f) => row[f]));
}

async function loadEverything() {
  const cfgResp = await fetch("models/config.json");
  config = await cfgResp.json();

  buildPntInputs();
  buildResultsTable();

  const statusEl = document.getElementById("status");
  let loaded = 0;
  const total = config.target_names.length;

  await Promise.all(
    config.target_names.map(async (target) => {
      const path = `models/${safeName(target)}.onnx`;
      const session = await ort.InferenceSession.create(path, {
        executionProviders: ["wasm"],
      });
      sessions[target] = session;
      loaded += 1;
      statusEl.textContent = `Loading models… (${loaded}/${total})`;
    })
  );

  statusEl.textContent = "Models loaded. Enter PNT counts and click Predict.";
  document.getElementById("btn-predict").disabled = false;
  document.getElementById("btn-predict").textContent = "Predict";
}

function buildPntInputs() {
  const container = document.getElementById("pnt-inputs");
  PNT_FEATURES.forEach((feat) => {
    const row = document.createElement("div");
    row.className = "field-row";
    row.innerHTML = `
      <label for="pnt-${feat}">${PNT_LABELS[feat]}</label>
      <input type="number" id="pnt-${feat}" min="0" max="175" value="0">
    `;
    container.appendChild(row);
  });
}

function buildResultsTable() {
  const tbody = document.getElementById("results-body");
  tbody.innerHTML = "";
  config.target_names.forEach((target) => {
    const tr = document.createElement("tr");
    tr.dataset.target = target;
    const maxVal = config.target_max[target];
    const r2 = config.target_cv_r2[target];
    const modelType = config.target_model_type[target];

    tr.innerHTML = `
      <td class="col-name">
        ${target}
        <span class="model-tag">[${modelType}]</span>
      </td>
      <td class="col-score"><span class="score-val">—</span></td>
      <td class="col-max">${maxVal}</td>
      <td class="col-r2">${r2.toFixed(3)}</td>
      <td class="col-bar">
        <div class="bar-track">
          <div class="bar-fill"></div>
          <div class="bar-label"></div>
        </div>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function colorForPct(pct) {
  if (pct >= 0.70) return getComputedStyle(document.documentElement).getPropertyValue("--high").trim();
  if (pct >= 0.40) return getComputedStyle(document.documentElement).getPropertyValue("--med").trim();
  return getComputedStyle(document.documentElement).getPropertyValue("--low").trim();
}

function updateResultsUI() {
  config.target_names.forEach((target) => {
    const tr = document.querySelector(`#results-body tr[data-target="${CSS.escape(target)}"]`);
    if (!tr) return;
    const scoreEl = tr.querySelector(".score-val");
    const fill = tr.querySelector(".bar-fill");
    const label = tr.querySelector(".bar-label");

    const score = lastPreds ? lastPreds[target] : null;
    const maxVal = config.target_max[target];

    if (score == null) {
      scoreEl.textContent = "—";
      scoreEl.style.color = "";
      fill.style.width = "0%";
      label.textContent = "";
      return;
    }

    const pct = maxVal > 0 ? score / maxVal : 0;
    const color = colorForPct(pct);

    scoreEl.textContent = score.toFixed(2);
    scoreEl.style.color = color;
    fill.style.width = `${Math.max(0, Math.min(100, pct * 100))}%`;
    fill.style.background = color;
    label.textContent = `${Math.round(pct * 100)}%`;
    label.style.color = pct > 0.5 ? "white" : "#212121";
  });
}

async function runPrediction() {
  const pntVals = {};
  let total = 0;
  for (const feat of PNT_FEATURES) {
    const v = parseInt(document.getElementById(`pnt-${feat}`).value, 10) || 0;
    pntVals[feat] = v;
    total += v;
  }

  if (total === 0) {
    alert("All PNT counts are 0. Please enter at least one value.");
    return;
  }

  const row = engineerFeatureRow(pntVals);
  const featArray = buildFeatureArray(row, config.feature_names);

  const preds = {};
  for (const target of config.target_names) {
    const session = sessions[target];
    const inputName = session.inputNames[0];
    const tensor = new ort.Tensor("float32", featArray, [1, featArray.length]);
    const results = await session.run({ [inputName]: tensor });
    const outputName = session.outputNames[0];
    const raw = Number(results[outputName].data[0]);
    const maxVal = config.target_max[target];
    preds[target] = Math.max(0, Math.min(maxVal, raw));
  }

  lastPreds = preds;
  updateResultsUI();

  const statusEl = document.getElementById("status");
  statusEl.textContent = `Predicted. Correct=${pntVals.pntCorrect}, Total=${total}`;
  document.getElementById("btn-export").disabled = false;
}

function clearAll() {
  PNT_FEATURES.forEach((feat) => {
    document.getElementById(`pnt-${feat}`).value = 0;
  });
  lastPreds = null;
  updateResultsUI();
  document.getElementById("status").textContent = "Cleared. Enter PNT counts and click Predict.";
  document.getElementById("btn-export").disabled = true;
}

function exportCsv() {
  if (!lastPreds) return;

  const lines = [];
  lines.push(["WAB Subscore", "Predicted Score", "Max Score", "% of Max", "CV R2", "Model"].join(","));
  config.target_names.forEach((target) => {
    const score = lastPreds[target];
    const maxVal = config.target_max[target];
    const pct = ((score / maxVal) * 100).toFixed(1) + "%";
    lines.push([
      `"${target}"`, score.toFixed(3), maxVal, pct,
      config.target_cv_r2[target], config.target_model_type[target],
    ].join(","));
  });
  lines.push("");
  lines.push("--- Inputs ---");
  PNT_FEATURES.forEach((feat) => {
    lines.push([`"${PNT_LABELS[feat]}"`, document.getElementById(`pnt-${feat}`).value].join(","));
  });

  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "wab_predictions.csv";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

document.getElementById("btn-predict").addEventListener("click", runPrediction);
document.getElementById("btn-clear").addEventListener("click", clearAll);
document.getElementById("btn-export").addEventListener("click", exportCsv);

loadEverything().catch((err) => {
  console.error(err);
  document.getElementById("status").textContent =
    "Failed to load models: " + err.message;
});
