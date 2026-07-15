# renderer.py

import json
import urllib.parse

import config
import core.analyzer as analyzer


# 피그마 'Spiner With BG' 컴포넌트(Property 1=Default)에서 그대로 추출한 SVG 원본.
# 밝은 배경 링(opacity 0.3) + 진한 회전 아크 조합 — CSS로 rotate 애니메이션만 걸면
# 피그마의 다중 프레임 스핀 애니메이션과 동일한 시각 효과를 재현한다.
_SPINNER_ICON_SVG = (
    '<svg width="28" height="28" viewBox="0 0 28 28" fill="none" '
    'xmlns="http://www.w3.org/2000/svg">'
    '<path opacity="0.3" d="M28 14C28 21.732 21.732 28 14 28C6.26801 28 0 21.732 0 14'
    'C0 6.26801 6.26801 0 14 0C21.732 0 28 6.26801 28 14ZM4.2 14C4.2 19.4124 8.58761 23.8'
    ' 14 23.8C19.4124 23.8 23.8 19.4124 23.8 14C23.8 8.58761 19.4124 4.2 14 4.2'
    'C8.58761 4.2 4.2 8.58761 4.2 14Z" fill="#2B7FFF"/>'
    '<path d="M25.9 14C27.0598 14 28.0161 13.0546 27.8428 11.9079C27.6737 10.7891 '
    '27.3693 9.69249 26.9343 8.64243C26.2307 6.94387 25.1995 5.40053 23.8995 4.1005'
    'C22.5995 2.80048 21.0561 1.76925 19.3576 1.06569C18.3075 0.63074 17.2109 0.326285 '
    '16.0921 0.157207C14.9454 -0.0161134 14 0.940202 14 2.1C14 3.2598 14.9507 4.17751 '
    '16.084 4.42414C16.6525 4.54787 17.2102 4.72226 17.7503 4.94598C18.9393 5.43848 '
    '20.0196 6.16034 20.9296 7.07035C21.8397 7.98037 22.5615 9.06071 23.054 10.2497'
    'C23.2777 10.7898 23.4521 11.3475 23.5759 11.916C23.8225 13.0493 24.7402 14 25.9 14Z" '
    'fill="#2B7FFF"/></svg>'
)
_SPINNER_ICON_DATA_URI = "data:image/svg+xml," + urllib.parse.quote(_SPINNER_ICON_SVG)


# ============================================================================
# 앱 공통 CSS
# ============================================================================
def get_app_styles() -> str:
    css = """
<style>
/* 피그마 지정 폰트: Pretendard(본문 전반) + Plus Jakarta Sans(메인 헤드라인 전용) */
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

.stApp { background-color: #f8f9fa; }

.header-bar {
    display: flex; justify-content: space-between; align-items: center;
    padding: 14px 40px; background: #fff;
    border-bottom: 1px solid #e9ecef;
}
.header-bi   { font-size: 1.25rem; font-weight: 800; color: #2563eb; letter-spacing: -.5px; }
.header-name { font-size: 16px; font-weight: 500; color: #434655; }

.main-title-wrap { text-align: center; padding: 48px 16px 28px; }
.main-sub  { font-size: 18px; font-weight: 600; color: #2b7fff; margin-bottom: 6px; }
.main-head {
    font-family: 'Plus Jakarta Sans', 'Pretendard', sans-serif !important;
    font-size: 34px; font-weight: 700; color: #101828; line-height: 1.35;
}

.char-count { font-size: 0.78rem; color: #9ca3af; text-align: right; margin-top: 2px; }
.example-label {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 1.0rem;
  font-weight: 600;
  color: #6b7280;
  margin: 20px 0 10px;
}
.example-label::before,
.example-label::after {
  content: "";
  flex: 1;
  height: 1px;
  background: #e5e7eb;
}

.result-sentence { font-size: 18px; font-weight: 700; color: #191b23; margin-bottom: 4px; }
.result-roman    { font-size: 14px; color: #888888; font-style: italic; margin-bottom: 12px; }
.sent-counter    { font-size: 0.85rem; color: #9ca3af; text-align: right; }
.result-trans {
    font-size: 1.1rem; color: #374151; font-style: italic;
    margin-top: 16px; padding-top: 14px; border-top: 1px solid #f3f4f6;
    text-align: center;
}

/* 분석 중 로딩 스피너 — 피그마 'Spiner With BG' 아이콘(실제 SVG 추출)으로 교체 + 가운데 정렬 */
/* st.spinner()의 stElementContainer는 기본적으로 콘텐츠 폭만큼만 hug되므로(width:100%가
   아님) 자식(stSpinner)에 width:100%를 줘도 중앙 정렬 기준 폭 자체가 없다 — 부모부터
   먼저 폭을 채워줘야 justify-content:center가 실제로 화면 가운데를 기준으로 동작한다. */
div[data-testid="stElementContainer"]:has(> div[data-testid="stSpinner"]) {
    width: 100% !important;
    margin-top: 20px !important;
}
div[data-testid="stSpinner"] {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
}
/* stSpinner의 실제 아이콘+텍스트 행은 testid 없는 내부 wrapper div(해시 클래스라 이름을
   특정할 수 없음) 하나뿐이며, 그 div 자체가 width:100%로 부모를 꽉 채워버려서
   부모(stSpinner)의 justify-content:center가 무력화된다 — 이 wrapper 자신에게
   다시 한번 center 정렬을 줘야 아이콘+텍스트 묶음이 실제로 가운데로 모인다. */
div[data-testid="stSpinner"] > div {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
}
/* 원래 아이콘 내부 요소(svg 등)를 깊이 상관없이 전부 숨김 — 겹쳐 보이는 문제 방지 */
[data-testid="stSpinnerIcon"] * {
    display: none !important;
    visibility: hidden !important;
}
[data-testid="stSpinnerIcon"] {
    width: 28px !important;
    height: 28px !important;
    margin-right: 12px !important;
    background-image: url("PYTHON_SPINNER_DATA_URI") !important;
    background-size: 28px 28px !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    animation: figma-spin 0.9s linear infinite;
}
@keyframes figma-spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
div[data-testid="stSpinner"] p {
    font-size: 22px !important;
    font-weight: 500 !important;
    color: #4a5565 !important;
    margin: 0 0 0 12px !important;
}

/* 예시 카드 버튼 */
div[data-testid="column"] .stButton > button {
    background: #fff !important; border: 1px solid #e5e7eb !important;
    border-radius: 10px !important; color: #374151 !important;
    font-size: 0.86rem !important; padding: 12px 14px !important;
    width: 100% !important; text-align: left !important;
    white-space: pre-wrap !important; height: auto !important; min-height: 58px !important;
    line-height: 1.5 !important;
}
div[data-testid="column"] .stButton > button:hover {
    border-color: #2563eb !important; color: #2563eb !important;
}

/* 분석하기 버튼 (type="primary") */
button[data-testid="stBaseButton-primary"] {
    background: #2563eb !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 0.95rem !important;
    padding: 10px 0 !important;
    transition: background .15s, box-shadow .15s !important;
}
button[data-testid="stBaseButton-primary"]:hover:not(:disabled) {
    background: #1d4ed8 !important;
    box-shadow: 0 2px 8px rgba(37,99,235,.35) !important;
}
button[data-testid="stBaseButton-primary"]:active:not(:disabled) {
    background: #1e40af !important;
    box-shadow: none !important;
}
button[data-testid="stBaseButton-primary"]:disabled {
    background: #d1d5db !important; color: #9ca3af !important;
    cursor: not-allowed !important;
}

/* 네비게이션 / 처음으로 버튼 */
.nav-wrap .stButton > button {
    background: #f3f4f6 !important; color: #374151 !important;
    border: 1px solid #e5e7eb !important; border-radius: 8px !important;
    font-weight: 500 !important;
}

#MainMenu, footer, header { visibility: hidden; }
/* 기존 max-width:1200px가 넓은 화면에서 스트림릿 wide 레이아웃 기본 좌우 패딩(80px)
   위에 또 여백을 더 얹어(양쪽 총 236px) 결과 시각화의 가로 스크롤을 불필요하게
   유발했다 — 피그마 '분석 결과' 화면(1920px 캔버스에서 콘텐츠 폭 1552px, 약 80.8%)
   비율에 맞춰 상한을 넉넉히 늘려 일반적인 데스크톱 폭에서는 사실상 제한이 걸리지
   않도록 한다(초대형 모니터에서 줄 길이가 과도해지는 것만 방지). */
.block-container { padding-top: 0 !important; max-width: 1600px; }

/* 텍스트 입력 focus 테두리 색상 */
.stTextArea textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 1px #2563eb !important;
}

</style>
"""
    return css.replace("PYTHON_SPINNER_DATA_URI", _SPINNER_ICON_DATA_URI)


def render_header() -> str:
    return """
<div class="header-bar">
    <span class="header-name">한국어 문장 구조 분석 엔진</span>
</div>
"""


def render_title() -> str:
    return """
<div class="main-title-wrap">
    <div class="main-sub">한국어 문장 구조 분석 엔진</div>
    <div class="main-head">입력한 문장의 문법 요소와 구조를 세밀하게 분석합니다</div>
</div>
"""


def render_translation(translation: str) -> str:
    if not translation:
        return ""
    return f'<div class="result-trans">🌐 {translation}</div>'


def render_tts_button(sentence: str) -> str:
    """TTS 버튼 — 재생/정지/완료 상태를 JS로 자체 관리"""
    safe = sentence.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
    return f"""
<style>
  body {{ margin:0; background:transparent; }}
  #tts-btn {{
    display: inline-flex; align-items: center; gap: 6px;
    background: #fff; border: 1px solid #e5e7eb;
    border-radius: 20px; font-size: 14px; font-weight: 500; color: #155dfc;
    padding: 5px 14px; cursor: pointer;
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    transition: background .15s, color .15s;
  }}
  #tts-btn:hover {{ background: #f3f4f6; color: #1249c4; }}
  #tts-btn.playing {{ background: #eff6ff; border-color: #bfdbfe; color: #155dfc; }}
</style>
<button id="tts-btn">▶ Full Sentence</button>
<script>
(function(){{
  var btn = document.getElementById('tts-btn');
  var playing = false;
  function setPlaying(v) {{
    playing = v;
    btn.textContent = v ? '■ Full Sentence' : '▶ Full Sentence';
    btn.className = v ? 'playing' : '';
  }}
  function speak() {{
    window.speechSynthesis.cancel();
    var utt = new SpeechSynthesisUtterance('{safe}');
    utt.lang = 'ko-KR';
    utt.onend = function() {{ setPlaying(false); }};
    utt.onerror = function() {{ setPlaying(false); }};
    window.speechSynthesis.speak(utt);
    setPlaying(true);
  }}
  btn.addEventListener('click', function() {{
    if (playing) {{ window.speechSynthesis.cancel(); setPlaying(false); }}
    else {{ speak(); }}
  }});
}})();
</script>
"""


# ============================================================================
# 상단 원문 문장 색상 (구/Phrase 단위) — main.py의 result-sentence 표시에 사용
# ============================================================================
def render_colored_sentence(text: str, morph_spans_off, spans) -> str:
    """원문 문장 전체 텍스트에 구(Phrase) 단위 색상을 입혀 HTML로 반환한다.
    절(MainC/SubC 등)은 별도 브라켓(render_phrase_bars)이 이미 담당하고 있고,
    절이 없는 단문에서도 색이 빠짐없이 보이도록 여기서는 구(config.PHRASES)만
    색칠 대상으로 삼는다 — 모든 형태소는 어떤 구엔 반드시 속하므로 항상 색이 뜬다.
    render_phrase_bars와는 독립적인 함수(팔레트 값만 동일하게 맞춤)."""
    import html as _html

    phrase_palette = {
        "TP":   "#f97316",
        "SP":   "#f59e0b",
        "OP":   "#a78bfa",
        "VP":   "#ef4444",
        "AdvP": "#38bdf8",
        "AdjP": "#84cc16",
        "NP":   "#14b8a6",
        "NPS":  "#0d9488",
        "CP":   "#6366f1",
    }

    phrase_spans = [t for t in spans if t[0] in config.PHRASES]
    span_len = lambda t: (t[2] - t[1])
    phrase_spans.sort(key=lambda x: (span_len(x), x[1], x[0]))

    token_colors = {}
    for label, s, e in phrase_spans:
        color = phrase_palette.get(label, "#94a3b8")
        for idx in range(s, e + 1):
            token_colors.setdefault(idx, color)

    char_color = {}
    for idx, (s, e) in enumerate(morph_spans_off):
        color = token_colors.get(idx)
        if not color:
            continue
        for ci in range(s, e):
            char_color[ci] = color

    parts = []
    cur_color = None
    buf = []

    def flush():
        if not buf:
            return
        seg = _html.escape("".join(buf))
        if cur_color:
            parts.append(f"<span style='color:{cur_color}'>{seg}</span>")
        else:
            parts.append(seg)
        buf.clear()

    for i, ch in enumerate(text):
        color = char_color.get(i)
        if color != cur_color:
            flush()
            cur_color = color
        buf.append(ch)
    flush()
    return "".join(parts)


# ============================================================================
# HTML Rendering
# ============================================================================
def render_phrase_bars(
    words,
    spans,
    groups,
    gloss=None,
    pos_tags=None,
    compound_spans=None,
    show_eojeol=True,
    show_pos=True,
    theme: str = "dark",
    density: str = "cozy",
    scale: float = 1.2,
):
    import html as _html
    import re

    N = len(words)

    phrases = [t for t in spans if t[0] in config.PHRASES]
    clauses = [t for t in spans if t[0] in config.CLAUSES]
    span_len = lambda t: (t[2] - t[1])

    phrases.sort(key=lambda x: (span_len(x), x[1], x[0]))
    clauses.sort(key=lambda x: (span_len(x), x[1], x[0]))

    levels = analyzer.pack_levels(phrases) + analyzer.pack_levels(clauses)

    # 키: SpanLabels 식별자, 값: 색상
    palette = {
        "TP":         "#f97316",
        "SP":         "#f59e0b",
        "OP":         "#a78bfa",
        "VP":         "#ef4444",
        "AdvP":       "#38bdf8",
        "AdjP":       "#84cc16",
        "NP":         "#14b8a6",
        "NPS":        "#0d9488",
        "CP":         "#6366f1",
        "MainC":      "#22c55e",
        "SubC":       "#60a5fa",
        "Sentence":   "#22c55e",
        "QuoteC_Dir": "#f472b6",
        "QuoteC_Ind": "#fb7185",
        "EmC_Adj":    "#c084fc",
        "EmC_N":      "#a3e635",
        "EmC_Adv":    "#fb923c",
    }

    def color_for(label: str) -> str:
        return palette.get(label, "#94a3b8")

    # ── 형태소별 글로스 색상 맵 ───────────────────────────────────────────────
    # 미리내 벤치마킹: 구/절 브라켓뿐 아니라 글로스 텍스트 자체도 색으로 구분되도록.
    # 절(MainC/SubC/QuoteC 등)이 없는 단문에서도 색이 보이도록 구(Phrase)까지 포함해서
    # 가장 좁은(가장 안쪽) 스팬의 색을 사용한다. phrases/clauses는 각각 내부적으로만
    # 정렬돼 있으므로 합친 뒤 스팬 길이 기준으로 다시 정렬해야 전체에서 가장 작은
    # 스팬이 먼저 온다 — setdefault로 순회하면 그 가장 작은 스팬이 자동으로 우선한다.
    all_spans_by_size = sorted(phrases + clauses, key=lambda x: (span_len(x), x[1], x[0]))
    token_gloss_color = {}
    for label, s, e in all_spans_by_size:
        color = color_for(label)
        for idx in range(s, e + 1):
            token_gloss_color.setdefault(idx, color)

    # ── compound 구간 인덱스 맵 ───────────────────────────────────────────────
    # 각 형태소 인덱스 → 소속 compound 구간 (head_idx, end_idx) or None
    compound_map = {}
    if compound_spans:
        for item in compound_spans:
            cs, ce = (item["s"], item["e"]) if isinstance(item, dict) else item
            for ci in range(cs, ce + 1):
                compound_map[ci] = (cs, ce)

    # ── 토큰 행 ──────────────────────────────────────────────────────────────
    re_punct = re.compile(r"^[^\w\s]+$")
    # compound head 셀: gloss/pos 텍스트를 head에만 표시, tail은 비움
    # → JS overlay 없이 CSS grid가 높이를 자동 통일
    compound_heads = {item["s"]: item for item in (compound_spans or [])}

    tok_cells = []
    for i, w in enumerate(words):
        cspan = compound_map.get(i)
        cspan_attrs = ""
        if cspan is not None:
            cspan_attrs = f" data-cs='{cspan[0]}' data-ce='{cspan[1]}'"

        if cspan is not None:
            cs = cspan[0]
            if i == cs:
                # head: compound의 gloss/pos를 표시
                cd = compound_heads.get(cs, {})
                g_display = cd.get("gloss") or ""
                p_display = cd.get("pos_eng") or ""
            else:
                # tail: 비움 (min-height 유지용 제로폭 공백)
                g_display = "​"
                p_display = ""
        else:
            g = gloss[i] if (gloss and gloss[i] is not None) else ""
            if (not g) and re_punct.match(w):
                g = w
            if g == "":
                g = "​"
            g_display = g
            p_display = pos_tags[i] if (show_pos and pos_tags) else ""

        gloss_color = token_gloss_color.get(i)
        gloss_style = f" style='color:{gloss_color}'" if gloss_color else ""

        tok_cells.append(
            f"<div class='cell tokcell' data-idx='{i}'{cspan_attrs}>"
            f"  <div class='tok-surf'>{_html.escape(w)}</div>"
            f"  <div class='tok-gloss'{gloss_style}>{_html.escape(g_display)}</div>"
            f"  <div class='tok-pos'>{_html.escape(p_display)}</div>"
            "</div>"
        )
    tok_row_html = "".join(tok_cells)

    # ── 어절칩 ───────────────────────────────────────────────────────────────
    ej_row = analyzer.render_eojeol_row(groups, N) if groups else ""

    # ── bracket 데이터 (JS가 SVG로 렌더링) ──────────────────────────────────
    # 각 span의 s(시작 형태소 idx), e(끝 형태소 idx), level, label, color를
    # JSON으로 내려보내면 JS가 .tokcell의 실제 픽셀 중앙을 읽어 SVG를 그림.
    bracket_data = []
    for level_idx, level in enumerate(levels):
        for label, s, e in level:
            bracket_data.append({
                "s": s, "e": e,
                "label": config.SPAN_DISPLAY.get(label, label),
                "color": color_for(label),
                "level": level_idx,
            })
            
    bracket_json = json.dumps(bracket_data, ensure_ascii=False)
    # compound_json: main.py에서 {s, e, gloss, pos_eng} dict 리스트로 전달됨
    compound_json = json.dumps(compound_spans or [], ensure_ascii=False)

    # level 수 기준으로 SVG 영역 높이 사전 계산
    n_levels = len(levels)
    level_h  = 48   # 레벨당 높이 (px): 가로선 간격 (브라켓 그룹 위아래 여백 6px씩 포함)
    svg_h    = max(n_levels * level_h + 24, 8)

    # 좌우 패딩을 20px 24px → 16px 12px로 줄이고 min_col_px도 소폭 축소해
    # 결과 카드의 실제 가용 폭(main.py에서 block-container max-width를 넓힌 것과 함께)
    # 대비 불필요한 가로 스크롤 유발을 최소화한다. 24px pill 폰트가 들어갈 최소
    # 여유는 유지(100px 아래로는 2~3음절 단어 pill이 셀 경계를 넘어가기 시작함).
    min_col_px   = 100
    min_total_w  = N * min_col_px + 24

    html = f"""
<style>
  body {{ margin:0; }}
  .outer-scroll {{
    overflow-x: auto; overflow-y: hidden; width: 100%; padding-bottom: 6px;
    scrollbar-width: thin; scrollbar-color: #9ca3af transparent;
  }}
  .outer-scroll::-webkit-scrollbar {{ height: 8px; }}
  .outer-scroll::-webkit-scrollbar-track {{ background: transparent; }}
  .outer-scroll::-webkit-scrollbar-thumb {{ background: #9ca3af; border-radius: 4px; }}
  .container {{
    background:#fff; color:#111827; border-radius:12px; padding:16px 12px;
    min-width:{min_total_w}px; box-sizing:border-box;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    position: relative;
  }}
  /* 어절+토큰을 하나의 그리드로 통합 — 컬럼 너비 완전 공유 */
  .unified-grid {{
    display:grid;
    grid-template-columns:repeat({N}, minmax({min_col_px}px,1fr));
    grid-template-rows: auto auto;
    gap:0 0;
  }}
  .ejitem {{ grid-row:1; min-width:0; }}
  .cell   {{ min-width:0; box-sizing:border-box; grid-row:2; }}

  /* 어절칩 */
  .ejchip {{
    background:#f1f5f9; border:1px solid #cbd5e1; border-radius:8px;
    padding:7px 6px; text-align:center; font-weight:700; color:#1e293b;
    margin: 0 3px 10px; font-size:13px; white-space:nowrap;
    overflow:hidden; text-overflow:ellipsis;
  }}

  /* 형태소 토큰 */
  .tokcell {{
    display:flex; flex-direction:column; align-items:center;
    padding:0 2px; box-sizing:border-box;
  }}
  /* 개별 형태소 pill 박스 — 피그마 'chip B' 컴포넌트 스펙(단일 라인 유지, 줄바꿈 금지) */
  .tok-surf {{
    font-size:24px; font-weight:700; color:#4a5565;
    text-align:center; white-space:nowrap;
    background: #eff6ff; border-radius:999px;
    padding: 3px 10px; margin-bottom:4px;
    position: relative; z-index: 1;
  }}
  .tok-gloss {{ font-size:13px; font-weight:500; color:#6b7280; line-height:1.3; text-align:center; }}
  .tok-pos   {{ font-size:13px; font-weight:400; color:#888888; line-height:1.3; text-align:center; }}

  .tokcell.highlighted .tok-surf {{ background: #bfdbfe; }}

  /* compound 배너: 여러 형태소 tok-surf를 하나의 pill로 묶음 */
  .compound-banner {{
    position: absolute;
    background: #eff6ff;
    border: none;
    border-radius: 999px;
    pointer-events: none;
    z-index: 0;
  }}
  .compound-banner.highlighted {{ background: #bfdbfe; }}
  /* compound 소속 셀의 tok-surf는 배경 제거 — 배너가 대신함 */
  .tokcell[data-cs] .tok-surf {{
    background: transparent;
    position: relative;
    z-index: 1;
  }}
  /* compound gloss/pos는 head 셀에서 숨기고 절대위치 레이블로 대체 */
  .tokcell[data-cs] .tok-gloss,
  .tokcell[data-cs] .tok-pos {{ visibility: hidden; }}
  .compound-label {{
    position: absolute;
    pointer-events: none;
    z-index: 2;
    text-align: center;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .compound-label.gloss {{ font-size:10px; font-weight:400; color:#6b7280; line-height:1.3; }}
  .compound-label.pos   {{ font-size:9px;  font-weight:400; color:#9ca3af; line-height:1.3; }}

  /* SVG bracket 오버레이 */
  #bk-svg {{
    display:block; width:100%; height:{svg_h}px;
    margin-top:4px; overflow:visible;
  }}

  .tok-surf, .tok-gloss, .tok-pos {{
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }}
</style>
<div class="outer-scroll">
  <div class="container" id="vis-container">
    <div class="unified-grid" id="tok-grid">{ej_row}{tok_row_html}</div>
    <svg id="bk-svg"></svg>
  </div>
</div>
<script>
(function(){{
  var data   = {bracket_json};
  var cdata  = {compound_json};
  var levelH = {level_h};
  var svg    = document.getElementById('bk-svg');
  var grid   = document.getElementById('tok-grid');
  var cont   = document.getElementById('vis-container');

  function ns(tag, attrs) {{
    var el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.keys(attrs).forEach(function(k){{ el.setAttribute(k, attrs[k]); }});
    return el;
  }}

  function draw() {{
    svg.innerHTML = '';
    var cells = grid.querySelectorAll('.tokcell');

    // SVG 좌표계: SVG의 뷰포트 기준 left를 기준점으로 삼아
    // 각 셀의 중앙 x를 SVG 좌표계로 변환
    var svgRect = svg.getBoundingClientRect();

    var cellMidX = [];
    var cellBotY = 0;
    cells.forEach(function(c) {{
      var r = c.getBoundingClientRect();
      // SVG left를 기준으로 한 상대 x — SVG 좌표계와 1:1 대응
      cellMidX.push(r.left - svgRect.left + r.width / 2);
      var gridRect = grid.getBoundingClientRect();
      cellBotY = Math.max(cellBotY, r.bottom - gridRect.top);
    }});

    svg.style.marginTop = '0';

    var tickH   = 18;   // 수직 틱 높이 (px) — 가로선에서 위로 올라가는 길이
    var labelOY = 10;   // 가로선 아래 라벨까지 여백
    var marginY = 6;    // 브라켓 그룹(<g>) 위아래 여백 — 레벨끼리 겹치지 않도록 각 레벨 상단에서 밀어냄

    data.forEach(function(d) {{
      var s = d.s, e = d.e, lv = d.level;
      var x1 = cellMidX[s];
      var x2 = cellMidX[e];
      if(x1 === undefined || x2 === undefined) return;

      // 레벨 0 = 토큰 행 바로 아래, 레벨 올라갈수록 아래로
      var lineY = lv * levelH + tickH + marginY + 2;  // 가로선 y (svg 좌표)
      var tickTopY = lineY - tickH;           // 수직 틱 상단 = 가로선에서 위로


      // 브라켓 전체를 <g>로 묶음 — 이벤트를 그룹 단위로 처리해 깜빡임 방지
      var g = ns('g', {{ cursor: 'default' }});

      // 왼쪽 수직 틱
      g.appendChild(ns('line', {{
        x1: x1, y1: tickTopY, x2: x1, y2: lineY,
        stroke: d.color, 'stroke-width': 2, 'pointer-events': 'none'
      }}));
      // 오른쪽 수직 틱
      if(e > s) {{
        g.appendChild(ns('line', {{
          x1: x2, y1: tickTopY, x2: x2, y2: lineY,
          stroke: d.color, 'stroke-width': 2, 'pointer-events': 'none'
        }}));
      }}
      // 가로선
      g.appendChild(ns('line', {{
        x1: x1, y1: lineY, x2: x2, y2: lineY,
        stroke: d.color, 'stroke-width': 2, 'pointer-events': 'none'
      }}));

      // 라벨 (텍스트 → bbox 측정 → pill 배경 순서로 그림)
      var labelX = (x1 + x2) / 2;
      var labelY = lineY + labelOY + 11;
      var txt = ns('text', {{
        x: labelX, y: labelY,
        'text-anchor': 'middle',
        fill: d.color,
        'font-size': '11',
        'font-weight': '600',
        'font-family': "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        'pointer-events': 'none',
      }});
      txt.textContent = d.label;

      // getBBox로 pill 크기 계산: 텍스트를 임시로 svg에 붙여 측정 후 제거
      svg.appendChild(txt);
      var bb;
      try {{ bb = txt.getBBox(); }} catch(ex) {{ bb = {{x: labelX-20, y: labelY-11, width: 40, height: 13}}; }}
      svg.removeChild(txt);

      var pillPx = 6, pillPy = 3;
      var pillH  = bb.height + pillPy * 2;
      var pill   = ns('rect', {{
        x: bb.x - pillPx, y: bb.y - pillPy,
        width:  bb.width  + pillPx * 2,
        height: pillH,
        rx: pillH / 2, ry: pillH / 2,
        fill: d.color, opacity: '0.13',
        cursor: 'default',
        'pointer-events': 'all',
      }});
      g.appendChild(pill);
      g.appendChild(txt);

      // 호버 이벤트를 pill rect에만 부착 — 라벨 영역에만 반응
      (function(spanS, spanE) {{
        var active = false;
        function setHighlight(on) {{
          grid.querySelectorAll('.tokcell').forEach(function(c) {{
            var idx = parseInt(c.getAttribute('data-idx'));
            var cs = c.getAttribute('data-cs');
            var ce = c.getAttribute('data-ce');
            var inRange = (cs !== null)
              ? (parseInt(cs) >= spanS && parseInt(ce) <= spanE)
              : (idx >= spanS && idx <= spanE);
            if (inRange) {{ if(on) c.classList.add('highlighted'); else c.classList.remove('highlighted'); }}
            else if(on) c.classList.remove('highlighted');
          }});
          cont.querySelectorAll('.compound-banner').forEach(function(b) {{
            var bcs = parseInt(b.getAttribute('data-cs'));
            var bce = parseInt(b.getAttribute('data-ce'));
            if (bcs >= spanS && bce <= spanE) {{
              if(on) b.classList.add('highlighted'); else b.classList.remove('highlighted');
            }}
          }});
        }}
        pill.addEventListener('mouseenter', function() {{ if(active) return; active=true; setHighlight(true); }});
        pill.addEventListener('mouseleave', function() {{ active=false; setHighlight(false); }});
      }})(s, e);

      svg.appendChild(g);
    }});

    // compound 배너: tok-surf 행을 가로지르는 pill
    // offsetLeft/offsetTop 사용 — 브라우저 확대 배율에 무관한 CSS 레이아웃 좌표
    cont.querySelectorAll('.compound-banner').forEach(function(el){{ el.remove(); }});
    cont.querySelectorAll('.compound-label').forEach(function(el){{ el.remove(); }});
    if (cdata.length === 0) return;

    // el의 cont 기준 offset 좌표 계산 (offsetParent 체인)
    function offsetFrom(el, base) {{
      var top = 0, left = 0;
      var cur = el;
      while (cur && cur !== base) {{
        top  += cur.offsetTop;
        left += cur.offsetLeft;
        cur   = cur.offsetParent;
      }}
      return {{ top: top, left: left }};
    }}

    cdata.forEach(function(cd) {{
      var minLeft = Infinity, maxRight = -Infinity;
      var surfTop = null, surfBot = null;
      for (var ii = cd.s; ii <= cd.e; ii++) {{
        var cell = grid.querySelector('.tokcell[data-idx="' + ii + '"]');
        if (!cell) continue;
        var surf = cell.querySelector('.tok-surf');
        if (!surf) continue;
        var o = offsetFrom(surf, cont);
        if (surfTop === null) surfTop = o.top;
        surfBot  = o.top + surf.offsetHeight;
        minLeft  = Math.min(minLeft,  o.left);
        maxRight = Math.max(maxRight, o.left + surf.offsetWidth);
      }}
      if (surfTop === null) return;

      var banner = document.createElement('div');
      banner.className = 'compound-banner';
      banner.setAttribute('data-cs', cd.s);
      banner.setAttribute('data-ce', cd.e);
      banner.style.top    = surfTop + 'px';
      banner.style.left   = minLeft + 'px';
      banner.style.width  = (maxRight - minLeft) + 'px';
      banner.style.height = (surfBot - surfTop)  + 'px';
      cont.appendChild(banner);

      // compound gloss/pos: cont 기준 절대위치 레이블로 배너 너비에 맞춰 가운데 정렬
      // head 셀의 tok-gloss/tok-pos는 CSS로 visibility:hidden 처리
      var headCell = grid.querySelector('.tokcell[data-idx="' + cd.s + '"]');
      if (headCell) {{
        var compoundW = maxRight - minLeft;
        var glossEl = headCell.querySelector('.tok-gloss');
        var posEl   = headCell.querySelector('.tok-pos');

        // tok-gloss 실제 top을 cont 기준으로 직접 측정 — margin/padding 오차 없음
        var labelTop = glossEl ? offsetFrom(glossEl, cont).top : (surfBot + 4);

        if (glossEl && glossEl.textContent.trim()) {{
          var gLabel = document.createElement('div');
          gLabel.className = 'compound-label gloss';
          gLabel.textContent = glossEl.textContent;
          gLabel.style.left  = minLeft + 'px';
          gLabel.style.width = compoundW + 'px';
          gLabel.style.top   = labelTop + 'px';
          cont.appendChild(gLabel);  // DOM에 붙인 뒤 offsetHeight 측정
          labelTop += gLabel.offsetHeight || 14;
        }}
        if (posEl && posEl.textContent.trim()) {{
          var pLabel = document.createElement('div');
          pLabel.className = 'compound-label pos';
          pLabel.textContent = posEl.textContent;
          pLabel.style.left  = minLeft + 'px';
          pLabel.style.width = compoundW + 'px';
          pLabel.style.top   = labelTop + 'px';
          cont.appendChild(pLabel);
        }}
      }}
    }});


    // SVG 높이 재조정
    var needed = data.length > 0
      ? Math.max.apply(null, data.map(function(d){{ return d.level; }})) * levelH + levelH + 4
      : 8;
    svg.setAttribute('height', needed);
  }}

  // rAF 두 번 → 레이아웃 완료 후 draw (getBoundingClientRect 정확도 보장)
  function scheduleDraw() {{
    requestAnimationFrame(function() {{
      requestAnimationFrame(draw);
    }});
  }}
  if(document.readyState === 'complete') {{ scheduleDraw(); }}
  else {{ window.addEventListener('load', scheduleDraw); }}
  window.addEventListener('resize', draw);
}})();
</script>
"""
    # iframe 높이 = container padding(32) + ej-row(44) + tok-row(84) + svg + 스크롤바(20) + 여유(16)
    #   + gloss/pos 줄바꿈 여유(40) — .tok-gloss/.tok-pos는 white-space:nowrap이 없어서
    #   "Adjective-forming Suffix"처럼 긴 라벨은 100px 칸 안에서 2줄로 줄바꿈된다.
    #   84는 각 1줄 기준 높이라 그 경우 실제 행 높이가 더 커지므로, 세로 스크롤이
    #   생기지 않도록(가로 스크롤만 쓰도록) 넉넉히 더해준다.
    iframe_h = 32 + 44 + 84 + svg_h + 20 + 16 + 40
    return html, iframe_h
