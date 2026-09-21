"use strict";

const API_BASE = "/api";

const DOC_CLASSES = [
  {
    name: "Acórdão",
    desc: "Decisão colegiada proferida por tribunal.",
    color: "#1D9E75",
  },
  {
    name: "Recurso Extraordinário",
    desc: "Petição de recurso dirigida ao Supremo Tribunal Federal.",
    color: "#185FA5",
  },
  {
    name: "Agravo (ARE)",
    desc: "Impugnação ao recurso extraordinário inadmitido.",
    color: "#534AB7",
  },
  {
    name: "Despacho",
    desc: "Decisão interlocutória ou ordem judicial de andamento processual.",
    color: "#BA7517",
  },
  {
    name: "Sentença",
    desc: "Decisão judicial de primeira instância que encerra o processo.",
    color: "#993C1D",
  },
];

let activeTab = "upload";
let selectedFile = null;

function switchTab(tab) {
  activeTab = tab;

  const uploadPanel = document.getElementById("upload-panel");
  const textPanel   = document.getElementById("text-panel");
  const btnUpload   = document.getElementById("tab-upload-btn");
  const btnText     = document.getElementById("tab-text-btn");

  if (tab === "upload") {
    uploadPanel.style.display = "block";
    textPanel.style.display   = "none";
    btnUpload.classList.add("active");
    btnUpload.setAttribute("aria-selected", "true");
    btnText.classList.remove("active");
    btnText.setAttribute("aria-selected", "false");
  } else {
    uploadPanel.style.display = "none";
    textPanel.style.display   = "block";
    btnText.classList.add("active");
    btnText.setAttribute("aria-selected", "true");
    btnUpload.classList.remove("active");
    btnUpload.setAttribute("aria-selected", "false");
  }

  hideResult();
}

function handleFile(input) {
  const file = input.files && input.files[0];
  if (!file) return;
  applyFile(file);
}

function applyFile(file) {
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    showError("Apenas arquivos .pdf são aceitos.");
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    showError("O arquivo excede o limite de 20 MB.");
    return;
  }

  selectedFile = file;

  const dropZone = document.getElementById("drop-zone");
  const fileTag  = document.getElementById("file-tag");
  const tagName  = document.getElementById("file-tag-name");

  tagName.textContent    = file.name;
  dropZone.style.display = "none";
  fileTag.style.display  = "inline-flex";

  hideResult();
}

function removeFile() {
  selectedFile = null;
  document.getElementById("file-input").value = "";
  document.getElementById("drop-zone").style.display = "block";
  document.getElementById("file-tag").style.display  = "none";
  hideResult();
}

function handleDragOver(event) {
  event.preventDefault();
  event.stopPropagation();
  document.getElementById("drop-zone").classList.add("drag-over");
}

function handleDragLeave(event) {
  event.preventDefault();
  event.stopPropagation();
  document.getElementById("drop-zone").classList.remove("drag-over");
}

function handleDrop(event) {
  event.preventDefault();
  event.stopPropagation();
  document.getElementById("drop-zone").classList.remove("drag-over");

  const file = event.dataTransfer.files && event.dataTransfer.files[0];
  if (file) applyFile(file);
}

function updateCount() {
  const text = document.getElementById("text-input").value;
  const countEl = document.getElementById("char-count");
  const warnEl  = document.getElementById("char-warn");

  countEl.textContent = text.length.toLocaleString("pt-BR");
  warnEl.style.display = text.length > 0 && text.length < 50 ? "inline" : "none";
}

function validateInput() {
  if (activeTab === "upload") {
    if (!selectedFile) {
      showError("Selecione um arquivo PDF antes de classificar.");
      return false;
    }
    return true;
  }

  const text = document.getElementById("text-input").value.trim();
  if (text.length < 50) {
    showError("O texto precisa ter ao menos 50 caracteres para classificação.");
    return false;
  }
  return true;
}

function classify() {
  if (!validateInput()) return;

  const btn = document.getElementById("classify-btn");
  setLoadingState(btn, true);
  hideResult();

  const formData = new FormData();

  if (activeTab === "upload") {
    formData.append("file", selectedFile);
  } else {
    formData.append("text", document.getElementById("text-input").value.trim());
  }

  fetch(`${API_BASE}/classify`, { method: "POST", body: formData })
    .then(res => {
      if (!res.ok) {
        return res.json().then(err => {
          throw new Error(err.detail || "Erro na requisição: " + res.status);
        });
      }
      return res.json();
    })
    .then(data => {
      renderResult(data.label, data.confidence, data.scores, data.extraction_method, data.requires_review);
      setLoadingState(btn, false);
    })
    .catch(err => {
      showError(err.message || "Não foi possível conectar ao servidor. Tente novamente.");
      setLoadingState(btn, false);
    });
}

const EXTRACTION_LABELS = {
  ocr: "OCR (Tesseract)",
  text_input: "Texto direto",
};

function renderResult(label, confidence, scores, extractionMethod, requiresReview = false) {
  const docClass = DOC_CLASSES.find(c => c.name === label);
  const review = requiresReview || !docClass;
  const pct      = Math.round(confidence * 100);

  document.getElementById("result-label").textContent = review ? "Não classificado" : docClass.name;
  document.getElementById("result-desc").textContent = review
    ? "Revisão manual necessária. Confira o documento antes de definir seu tipo documental."
    : docClass.desc;
  document.getElementById("conf-pct").textContent     = pct + "%";

  const methodEl = document.getElementById("extraction-method");
  if (methodEl && extractionMethod) {
    methodEl.textContent = EXTRACTION_LABELS[extractionMethod] || extractionMethod;
    methodEl.style.display = "inline-flex";
  }

  const confBar = document.getElementById("conf-bar");
  confBar.style.width = "0%";
  confBar.style.background =
    review ? "#993C1D" : confidence >= 0.75 ? "#1D9E75" :
    confidence >= 0.50 ? "#BA7517" : "#993C1D";

  const confWrap = document.getElementById("conf-bar-wrap");
  confWrap.setAttribute("aria-valuenow", pct);

  requestAnimationFrame(() => {
    setTimeout(() => { confBar.style.width = pct + "%"; }, 30);
  });

  const scoresSorted = [...scores].sort((a, b) => b.score - a.score);
  const listEl = document.getElementById("scores-list");
  listEl.innerHTML = "";

  scoresSorted.forEach(s => {
    const cls   = DOC_CLASSES.find(c => c.name === s.name) || {};
    const p     = Math.round(s.score * 100);
    const color = cls.color || "#929292";
    const isTop = !review && s.name === label;

    const row = document.createElement("div");
    row.className = "score-row";
    row.innerHTML = `
      <span class="score-name" style="${isTop ? "font-weight:600;color:var(--text-primary)" : ""}">
        ${s.name}
      </span>
      <div class="score-bar-bg">
        <div class="score-bar-fill" style="width:0%;background:${color}"></div>
      </div>
      <span class="score-val">${p}%</span>
    `;
    listEl.appendChild(row);

    requestAnimationFrame(() => {
      setTimeout(() => {
        row.querySelector(".score-bar-fill").style.width = p + "%";
      }, 80);
    });
  });

  const resultSection = document.getElementById("result-section");
  resultSection.style.display = "block";
  resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function hideResult() {
  document.getElementById("result-section").style.display = "none";
}

function setLoadingState(btn, loading) {
  if (loading) {
    btn.disabled = true;
    btn.innerHTML = `
      <i class="fa-solid fa-spinner fa-spin" aria-hidden="true"></i>
      Classificando…
    `;
  } else {
    btn.disabled = false;
    btn.innerHTML = `
      <i class="fa-solid fa-brain" aria-hidden="true"></i>
      Classificar documento
    `;
  }
}

function showError(message) {
  const existing = document.getElementById("inline-error");
  if (existing) existing.remove();

  const el = document.createElement("p");
  el.id = "inline-error";
  el.setAttribute("role", "alert");
  el.style.cssText = `
    font-size:13px; color:#993C1D; margin-top:10px;
    display:flex; align-items:center; gap:6px;
  `;
  el.innerHTML = `
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="2" aria-hidden="true">
      <circle cx="12" cy="12" r="10"/>
      <line x1="12" y1="8" x2="12" y2="12"/>
      <line x1="12" y1="16" x2="12.01" y2="16"/>
    </svg>
    ${message}
  `;

  const btn = document.getElementById("classify-btn");
  btn.parentNode.insertBefore(el, btn);

  setTimeout(() => { if (el.parentNode) el.remove(); }, 5000);
}

