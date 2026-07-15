// web/app.js
//
// main.py(Streamlit)가 하던 화면 조작(입력 → 로딩 → 결과 → 이전/다음 → 처음으로)을
// vanilla JS로 재작성한 것. 상태는 st.session_state 대신 이 파일의 변수가 들고 있고,
// st.rerun() 대신 필요한 DOM만 직접 갱신한다. 분석 자체는 /api/analyze 호출로
// api/pipeline.py(원본 run_analysis를 그대로 옮긴 것)가 수행한다.

const EXAMPLE_SENTENCES = [
  "나는 어제 도서관에서 책을 읽었다.",
  "그는 내일 비가 올 것이라고 말했다.",
  "작년 이맘때는 눈이 왔는데 올해는 벌써 덥다.",
];

const els = {
  inputText: document.getElementById("input-text"),
  charCount: document.getElementById("char-count"),
  analyzeBtn: document.getElementById("analyze-btn"),
  exampleRow: document.getElementById("example-row"),
  spinnerWrap: document.getElementById("spinner-wrap"),
  resultSection: document.getElementById("result-section"),
  resultIdx: document.getElementById("result-idx"),
  resultTotal: document.getElementById("result-total"),
  resultSentence: document.getElementById("result-sentence"),
  resultRoman: document.getElementById("result-roman"),
  ttsFrame: document.getElementById("tts-frame"),
  vizFrame: document.getElementById("viz-frame"),
  resultTrans: document.getElementById("result-trans"),
  resultTransText: document.getElementById("result-trans-text"),
  prevBtn: document.getElementById("prev-btn"),
  nextBtn: document.getElementById("next-btn"),
  resetBtn: document.getElementById("reset-btn"),
};

let results = [];
let currentIdx = 0;

// ── 예시 문장 버튼 ──────────────────────────────────────────────────────────
EXAMPLE_SENTENCES.forEach((sent) => {
  const btn = document.createElement("button");
  btn.className = "example-card";
  btn.type = "button";
  btn.textContent = sent;
  btn.addEventListener("click", () => {
    els.inputText.value = sent;
    onInputChange();
    results = [];
    els.resultSection.hidden = true;
  });
  els.exampleRow.appendChild(btn);
});

// ── 입력 ────────────────────────────────────────────────────────────────
function onInputChange() {
  const text = els.inputText.value;
  els.charCount.textContent = text.length;
  els.analyzeBtn.disabled = text.trim().length === 0;
}
els.inputText.addEventListener("input", onInputChange);

// ── 분석 실행 ───────────────────────────────────────────────────────────
els.analyzeBtn.addEventListener("click", async () => {
  const text = els.inputText.value.trim();
  if (!text) return;

  els.analyzeBtn.disabled = true;
  els.spinnerWrap.hidden = false;
  els.resultSection.hidden = true;

  try {
    const res = await fetch("api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error(`서버 오류 (${res.status})`);
    const data = await res.json();
    results = data.results;
    currentIdx = 0;
    renderResult();
    els.resultSection.hidden = false;
  } catch (err) {
    alert(`분석 중 오류가 발생했습니다: ${err.message}`);
  } finally {
    els.spinnerWrap.hidden = true;
    els.analyzeBtn.disabled = els.inputText.value.trim().length === 0;
  }
});

// ── 결과 렌더링 ─────────────────────────────────────────────────────────
function renderResult() {
  const total = results.length;
  const cur = results[currentIdx];

  els.resultIdx.textContent = currentIdx + 1;
  els.resultTotal.textContent = total;
  els.resultSentence.innerHTML = cur.sentence_html;
  els.resultRoman.textContent = cur.romanize || "";

  els.ttsFrame.srcdoc = cur.tts_html || "";

  els.vizFrame.srcdoc = cur.html || "";
  els.vizFrame.style.height = ((cur.html_h || 300) + 40) + "px";

  els.resultTrans.hidden = !cur.translation;
  els.resultTransText.textContent = cur.translation || "";

  els.prevBtn.style.display = total > 1 ? "" : "none";
  els.nextBtn.style.display = total > 1 ? "" : "none";
  els.prevBtn.disabled = currentIdx === 0;
  els.nextBtn.disabled = currentIdx === total - 1;
}

els.prevBtn.addEventListener("click", () => {
  if (currentIdx > 0) {
    currentIdx -= 1;
    renderResult();
  }
});
els.nextBtn.addEventListener("click", () => {
  if (currentIdx < results.length - 1) {
    currentIdx += 1;
    renderResult();
  }
});

// ── 처음으로 ────────────────────────────────────────────────────────────
els.resetBtn.addEventListener("click", () => {
  results = [];
  currentIdx = 0;
  els.inputText.value = "";
  onInputChange();
  els.resultSection.hidden = true;
});

onInputChange();
