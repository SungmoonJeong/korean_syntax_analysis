#!/usr/bin/env python3
"""
형태소 Span Extractor - KLUE-DP 관계 기반

KLUE-DP 관계 라벨 (세종 treebank 기반):
- NP_SBJ: 주어
- NP_OBJ: 목적어
- NP_MOD: 관형어 (N*의)
- NP_AJT: 부사어
- NP_CMP: 보어
- NP_CNJ: 접속
- VP, VP_MOD: 동사구
- VNP, VNP_MOD: 서술명사구
- AP: 부사구
- DP: 관형사
- morph: 어절 내 형태소 연결

품사 기반 규칙:
- JX: 보조사 (은/는) → Topic
- JKS: 주격조사 (이/가) → Subject
- JKO: 목적격조사 (을/를) → Object
- JKB: 부사격조사 (에/에서/로) → Adverbial
- JKG: 관형격조사 (의) → Adjectival (possessive)
- JKC: 보격조사 → Complement
- JKQ: 인용격조사 (라고/고) → Quotation
- ETM: 관형형전성어미 → Adjectival (V*+ETM)
- EC: 연결어미 → Subordinate
- EF: 종결어미 → Main clause
"""

from typing import List, Tuple, Set, Dict, Optional
from collections import defaultdict

from config import SpanLabels


class MorphemeSpanExtractorKLUE:
    """
    KLUE-DP 관계 기반 Span Extractor

    형태소 단위 입력을 받아서 구/절 추출
    """

    def __init__(self, tokens: List[str], xpos: List[str], arcs: List[int], rels: List[str]):
        """
        Args:
            tokens: 형태소 리스트 ['나', '는', '학교', '에', '간다', '.']
            xpos: 품사 태그 ['NP', 'JX', 'NNG', 'JKB', 'VV+EC', 'SF']
            arcs: head indices (1-indexed, 0=root)
            rels: KLUE-DP 관계 ['morph', 'NP_SBJ', 'morph', 'NP_AJT', 'VP', 'VP']
        """
        self.tokens = tokens
        self.xpos = xpos
        self.arcs = arcs
        self.rels = rels
        self.N = len(tokens)

        # 결과
        self.spans: List[Tuple[str, int, int]] = []

        # 의존 관계 그래프 (children)
        self.children: Dict[int, List[int]] = defaultdict(list)
        for i, h in enumerate(arcs):
            if h > 0:
                self.children[h - 1].append(i)

        # 종속절 정보
        self.subordinate_ranges: List[Tuple[int, int]] = []

        # 인용절 범위 (SubC 추출 시 내부 EC 제외용)
        self.quotation_ranges: List[Tuple[int, int]] = []

        # JKB(으로/에 등) + 동사+EC 부사어 동사구('인하여', '대하여' 등) 범위 — VP 추출 제외용
        self._aux_vp_after_jkb: Set[Tuple[int, int]] = set()

    def extract(self) -> List[Tuple[str, int, int]]:
        """Span 추출 메인"""
        self.spans = []

        # ROOT 찾기
        root_idx = self._find_root()
        if root_idx is None:
            return []

        # 0) JKB + 동사+EC 부사어 동사구('인하여', '대하여' 등) 사전 스캔 — VP 추출 제외용
        self._scan_aux_vp_after_jkb()

        # 0-1) NPS(명사 나열) 선행 추출 — SubC가 주어 어절을 NPS 전체로 확장하는 데 필요
        self._extract_noun_phrase_sequence()

        # 0-2) AdjP/EmC_Adj(관형절) 선행 추출 — SubC가 주어 어절의 관형절을 포함하는 데 필요
        self._extract_adjectival()

        # 1) 인용절 추출
        self._extract_quotations()

        # 2) 종속절 추출 (EC 기반)
        self._extract_subordinate_clauses()

        # 3) 주절 추출
        self._extract_main_clause(root_idx)

        # 4) 구문 추출
        self._extract_phrases()

        # 4-1) 나열(NPS) 첫 접속항 앞의 관형 수식어까지 NPS 범위 확장 (A안)
        self._extend_nps_with_leading_modifier()

        # 5) 정리 및 중복 제거
        return self._deduplicate()

    def _find_root(self) -> Optional[int]:
        """ROOT (head=0) 찾기"""
        for i, h in enumerate(self.arcs):
            if h == 0:
                return i
        return None

    # ========================================================================
    # 품사/관계 헬퍼 함수
    # ========================================================================

    def _has_tag(self, idx: int, tag_prefix: str) -> bool:
        """특정 품사 태그가 있는지 확인"""
        if idx < 0 or idx >= self.N:
            return False
        return self.xpos[idx].startswith(tag_prefix)

    def _has_any_tag(self, idx: int, prefixes: tuple) -> bool:
        """여러 품사 중 하나라도 있는지"""
        if idx < 0 or idx >= self.N:
            return False
        return self.xpos[idx].startswith(prefixes)

    def _is_josa(self, idx: int, josa_type: str) -> bool:
        """특정 조사 타입인지 확인"""
        return self._has_tag(idx, josa_type)

    def _is_verbal(self, idx: int) -> bool:
        """동사/형용사/보조용언인지 (XSV/XSA 포함: 공부하다/다양하다 등 파생 동사·형용사)"""
        return self._has_any_tag(idx, ('VV', 'VA', 'VX', 'VCN', 'VCP', 'XSV', 'XSA'))

    def _is_nominal(self, idx: int) -> bool:
        """체언(명사류)인지"""
        return self._has_any_tag(idx, ('NN', 'NP', 'NR', 'XSN'))

    def _find_josa_right(self, idx: int, josa_type: str, max_dist: int = 3) -> Optional[int]:
        """오른쪽에서 조사 찾기 (morph 관계로 연결된 것)"""
        for i in range(idx, min(idx + max_dist, self.N)):
            if self._is_josa(i, josa_type):
                return i
        return None

    def _find_josa_in_eojeol(self, start_idx: int, josa_type: str) -> Optional[int]:
        """어절 내에서 조사 찾기 (morph 관계 따라가기)"""
        # 현재 위치 확인
        if self._is_josa(start_idx, josa_type):
            return start_idx

        # 오른쪽으로 morph 관계 따라가기
        for i in range(start_idx + 1, self.N):
            if self.rels[i] == 'morph' and self.arcs[i] - 1 >= start_idx:
                if self._is_josa(i, josa_type):
                    return i
            elif self.rels[i] != 'morph':
                # 새 어절 시작
                break

        return None

    # ========================================================================
    # 어절/명사구 확장
    # ========================================================================

    def _expand_eojeol_left(self, josa_idx: int) -> int:
        """조사/핵어 위치에서 왼쪽으로 어절 시작점 찾기.
        SuPar 어절 내 형태소는 'rel=morph'를 가지며 어절 내 오른쪽 형태소를 head로 가리킴.
        즉 i의 rel==morph이고 arcs[i]-1이 i보다 크면 i는 어절 내 형태소.
        """
        start = josa_idx
        for i in range(josa_idx - 1, -1, -1):
            # i가 morph rel이고 head가 i보다 오른쪽(같은 어절)을 가리키면 포함
            if self.rels[i] == 'morph' and self.arcs[i] - 1 > i:
                start = i
                continue
            break
        return start

    def _expand_np_with_modifier(self, josa_idx: int) -> Tuple[int, int]:
        """
        명사구 확장: 조사 위치에서 왼쪽 명사 + 수식어까지 포함

        규칙: N*가 수식을 받는다면 거기까지 모두 포함
        VP_MOD(관형절)는 이미 추출된 AdjP를 참조하여 start 확장

        Args:
            josa_idx: 조사 위치

        """
        # 1. 어절 내 시작점
        eojeol_start = self._expand_eojeol_left(josa_idx)
        # JX 바로 앞이 JKB(에/에서/로 등)인 경우(처음엔 = 처음+에+는):
        # JKB도 같은 어절로 보고 그 앞 명사까지 확장
        if (self._has_tag(josa_idx, 'JX')
                and eojeol_start == josa_idx
                and josa_idx > 0
                and self._has_tag(josa_idx - 1, 'JKB')):
            eojeol_start = self._expand_eojeol_left(josa_idx - 1)
        start = eojeol_start

        # 2. 왼쪽으로 수식어 확장 (관형사, 복합명사 등). while 기반: start를 왼쪽 어절로
        # 점프시키면 i=start-1로 재동기화 (for range 순차 감소는 건너뛴 어절 내부에서 끊김)
        i = eojeol_start - 1
        while i >= 0:
            rel = self.rels[i]

            # NP_MOD (관형격 "의") — 조사 자체이므로 명사류 체크 없이 통과
            if rel == 'NP_MOD':
                start = self._expand_eojeol_left(i)
                i = start - 1
                continue

            # 관형절(ETM/XSV로 끝나는 어절) — 이미 추출된 AdjP가 있으면 그 시작까지 확장
            # (rel=VP_MOD가 아니어도 ETM이면 처리. DP/NP 분기보다 먼저 둬야 rel=NP인 ETM이 잡힘)
            if self._has_any_tag(i, ('ETM', 'XSV')):
                adj_match = [
                    (s, e) for lbl, s, e in self.spans
                    if lbl == SpanLabels.ADJP and s <= i <= e
                ]
                if adj_match:
                    adj_match.sort(key=lambda x: x[1] - x[0], reverse=True)
                    start = adj_match[0][0]
                    break  # 관형절 경계에서 왼쪽 탐색 종료
                elif rel in ('VP_MOD', 'VNP_MOD'):
                    # AdjP가 아직 추출되지 않았으면 ETM 어절(VNP_MOD subtree) 시작까지 역추적
                    etm_nodes = self._get_subtree(i)
                    etm_nodes = {n for n in etm_nodes if n <= i}
                    if etm_nodes:
                        start = min(etm_nodes)
                    break  # 관형절 경계에서 왼쪽 탐색 종료
                # ETM이지만 확장 근거(AdjP 매칭/VP_MOD)가 없으면 break하지 않고
                # 아래 일반 분기(복합명사 morph 등)에 맡긴다

            # DP (관형사), NP (복합명사) — 명사/관형사류여야 함
            # (조사가 NP rel을 잘못 갖는 경우(예: 'ᆫ'(JX, rel=NP))를 걸러내기 위함)
            if rel in {'DP', 'NP'}:
                if not (self._is_nominal(i) or self._has_any_tag(i, ('MM', 'SL', 'SN', 'XPN', 'XSN'))):
                    break
                # MM(관형사)을 포함하는 AdjP가 이미 있으면 AdjP 시작으로 확장 (예: '서로 다른 선택을')
                if self._has_any_tag(i, ('MM', 'MMA', 'MMD', 'MMN')):
                    adj_match = [
                        (s, e) for lbl, s, e in self.spans
                        if lbl == SpanLabels.ADJP and s <= i <= e
                    ]
                    if adj_match:
                        adj_match.sort(key=lambda x: x[1] - x[0], reverse=True)
                        start = adj_match[0][0]
                        i = start - 1
                        continue
                start = self._expand_eojeol_left(i)
                i = start - 1
                continue

            # 관형사(MM)
            if self._has_any_tag(i, ('MM', 'MMA', 'MMD', 'MMN')):
                start = i
                i -= 1
                continue

            # 명사류 연속 (복합명사)
            if self._is_nominal(i) and self.rels[i] == 'NP':
                start = i
                i -= 1
                continue

            # 그 외는 중단
            break

        return (start, josa_idx)

    def _extend_start_with_nps(self, s: int) -> int:
        """s가 이미 추출된 NPS(명사 나열) 범위 안에 있으면 그 NPS의 시작점으로 확장.
        예: '문수원기비와 국보였던 극락전 등이' — NPS[문수원기비와...등]이 있으면
        SP/OP/CP/AdvP 시작점을 NPS 시작으로 확장한다.
        """
        nps_match = [
            (ns, ne) for lbl, ns, ne in self.spans
            if lbl == SpanLabels.NPS and ns <= s and ne >= s
        ]
        if nps_match:
            return min(ns for ns, ne in nps_match)
        return s

    def _extend_start_with_adjp(self, s: int) -> int:
        """s 바로 왼쪽에 끝나는 관형절(AdjP/EmC_Adj)이 인접하면 그 시작점으로 확장.
        예: '신고 계신 안전화가' — 주어 어절 시작 s='안전화'(4) 바로 왼쪽에서
        AdjP[신고 계신](0~3)이 끝나면 s를 0으로 당긴다. 관형절이 연쇄로 겹치면
        반복 적용한다(`작은 빨간 공`처럼 관형절이 여러 개).

        SubC가 주어 어절의 관형절을 절 범위에 포함시키는 데 쓴다(AdjP는 SubC보다
        먼저 선행 추출됨). _extend_start_with_nps의 '인접' 버전.
        """
        while True:
            adj_match = [
                ns for lbl, ns, ne in self.spans
                if lbl in (SpanLabels.ADJP, SpanLabels.EMC_ADJ) and ne == s - 1
            ]
            if not adj_match:
                return s
            s = min(adj_match)

    def _find_clause_start(self, verb_idx: int) -> int:
        """동사 위치에서 절의 시작점 찾기 (subtree)"""
        nodes = self._get_subtree(verb_idx)
        return min(nodes) if nodes else verb_idx

    def _get_subtree(self, root_idx: int) -> Set[int]:
        """subtree 노드 수집"""
        result = {root_idx}
        stack = [root_idx]

        while stack:
            u = stack.pop()
            for v in self.children[u]:
                if v not in result:
                    result.add(v)
                    stack.append(v)

        return result

    def _has_internal_arguments(self, _verb_idx, start: int, end: int) -> bool:
        """절 범위[start:end] 안에 논항(주어/목적어/부사어 역할)이 있는지 확인.
        논항이 있으면 절(EmC_Adj/EmC_N), 없으면 단순 구(AdjP/NP).

        SuPar는 대등 접속 구조에서 공유 논항을 첫 번째 서술어의 head로 붙이는 경향이
        있으므로, 범위 내 어떤 노드든 논항 rel이면 인정한다.
        """
        _ARG_RELS = {'NP_SBJ', 'NP_OBJ', 'NP_AJT', 'NP_CMP', 'NP_CNJ', 'AP'}
        for n in range(start, end + 1):
            if self.rels[n] in _ARG_RELS:
                return True
        return False

    # ========================================================================
    # Clause 추출
    # ========================================================================

    # 형태소 분석기가 JKQ 대신 EC로 분석하는 간접인용 어미 목록
    # '고'는 연결어미(싫고, 좋고 등)와 겹치므로 제외
    _INDIRECT_QUOTE_EC = {'라고', '다고', '냐고', '자고', '라며', '이라고', '이라며'}

    def _is_quote_trigger(self, i: int) -> bool:
        """인용절 트리거 형태소 여부 (JKQ 또는 간접인용 EC)"""
        if self._has_tag(i, 'JKQ'):
            return True
        if self._has_tag(i, 'EC') and self.tokens[i] in self._INDIRECT_QUOTE_EC:
            return True
        return False

    def _extract_quotations(self):
        """인용절 추출 (JKQ 또는 간접인용 EC: 라고/다고/냐고/자고/라며/고)"""
        for i in range(self.N):
            if not self._is_quote_trigger(i):
                continue

            quote_end = i

            # 직접 인용: 따옴표 찾기
            is_direct = False
            for j in range(i - 1, -1, -1):
                if self.tokens[j] in {'"', "'", '"', '"', "'", "'", '「', '」'}:
                    is_direct = True
                    break

            # 인용절 시작점: 트리거(라고 등) 어절의 morph chain 시작점부터
            # 트리거 어절의 맨 앞(morph chain 역추적)을 구함
            trigger_eojeol_start = i
            while trigger_eojeol_start > 0 and self.rels[trigger_eojeol_start - 1] == 'morph':
                trigger_eojeol_start -= 1

            # 트리거 직전 어절(morph chain)을 역추적해 인용절 내부 핵어를 찾음
            anchor = trigger_eojeol_start - 1
            while anchor >= 0 and self.rels[anchor] == 'morph':
                anchor -= 1
            if anchor >= 0:
                anchor_head = self.arcs[anchor]
                if anchor_head > 0:
                    candidate = anchor_head - 1
                    # anchor의 head가 트리거 어절 범위 밖(주절)을 가리키면
                    # 트리거 어절 시작 자체를 verb_idx로 사용
                    if candidate < trigger_eojeol_start or candidate > i:
                        verb_idx = trigger_eojeol_start
                    else:
                        verb_idx = candidate
                else:
                    verb_idx = anchor
                subtree = self._get_subtree(verb_idx)
                subtree.add(verb_idx)
                # 인용절은 트리거 이전 범위만 포함
                nodes_before = [n for n in subtree if n < i]
                quote_start = min(nodes_before) if nodes_before else trigger_eojeol_start
            else:
                quote_start = max(0, i - 5)

            label = SpanLabels.QUOTEC_DIR if is_direct else SpanLabels.QUOTEC_IND
            self.spans.append((label, quote_start, quote_end))
            self.quotation_ranges.append((quote_start, quote_end))

    def _extract_subordinate_clauses(self):
        """종속절 추출 (EC 연결어미 기반)"""
        self.subordinate_ranges = []

        for i in range(self.N):
            # EC (연결어미) 찾기
            if not self._has_tag(i, 'EC'):
                continue

            # 인용절 내부의 EC는 SubC로 내지 않음 (인용절로 이미 처리됨)
            if any(qs <= i <= qe for qs, qe in self.quotation_ranges):
                continue

            # rel=VP인 EC는 종속절 연결어미 화이트리스트에 있을 때만 SubC로 처리
            # (고/며/아·어 등 대등 연결어미는 제외 — VP로 이어붙이며 SubC 경계로 삼지 않음)
            _SUBC_EC = {
                '는데', '은데', 'ㄴ데',       # 대립/배경
                '니까', '으니까', '니',        # 이유/원인
                '면서', '으면서',              # 동시
                '면', '으면',                  # 조건
                '라면', '이라면',              # 조건 ((이)라면 = 지정사 어간+면, '아니라면' 등)
                '아서', '어서', '서',          # 이유/순서
                '고서', '아도', '어도',        # 양보/순서
                '지만', '만',                  # 대조
                '라', '이라', '아니라', '이 아니라',  # 대조/전환 (아니라/이라)
                '때문에', '기 때문에',
                '므로', '으므로',              # 이유/원인
                '거나', '든가', '든지',        # 선택/나열
            }
            if self.rels[i] == 'VP_AJT':
                continue  # 부사절 형성 EC(이렇게, 명확하게 등) → SubC 아님
            if self.rels[i] == 'VP' and self.tokens[i] not in _SUBC_EC:
                continue
            # 보조용언 연결 EC(아/어 + 지다/있다/없다 등)는 같은 VP 내부 연결이므로
            # 그 자체로 별도 SubC 경계가 되지 않음 (불타 + 아 + 없어 + 지 + 었 + 으며)
            if (self.tokens[i] in ('아', '어')
                    and i + 1 < self.N
                    and self.rels[i + 1] == 'morph'
                    and (self.tokens[i + 1] in self._AUX_VERB_AFTER_EC
                         or self._has_any_tag(i + 1, ('VX', 'XSV')))):
                continue

            # EC 뒤에 후속 용언/절이 없고 문말 보조사(요 등)만 남아 끝나면
            # 종속절이 아니라 구어체 문장 종결(는데요/어서요 등) → SubC 아님
            has_following_clause = False
            for j in range(i + 1, self.N):
                if self._is_verbal(j):
                    has_following_clause = True
                    break
                if self._has_any_tag(j, ('SF', 'SP', 'SE', 'SS')):
                    continue
                if self._has_tag(j, 'JX') and self.rels[j] == 'morph':
                    continue
                has_following_clause = True
                break
            if not has_following_clause:
                continue

            # EC 앞의 동사 찾기 (morph 관계 따라가기)
            # 보조용언 연결(-아/어 + 지다/있다/없다 등)이면 본용언까지 계속 탐색
            verb_idx = None
            for j in range(i - 1, -1, -1):
                if self._is_verbal(j):
                    verb_idx = j
                    if (j > 0
                            and self.tokens[j] in self._AUX_VERB_AFTER_EC
                            and self.tokens[j - 1] in ('아', '어')):
                        continue  # 보조용언 → 그 앞의 '아/어'를 거쳐 본용언까지 계속
                    break
                # morph 관계거나 어미(E)/접미사(X)면 계속
                if self.rels[j] == 'morph' or self._has_any_tag(j, ('E', 'X')):
                    continue
                break

            if verb_idx is None:
                # 동사 없으면 EC 자체가 동사에 붙은 경우
                # VP 관계 찾기
                for j in range(i - 1, -1, -1):
                    if self.rels[j] == 'VP' or self._is_verbal(j):
                        verb_idx = j
                        break
                    if self.rels[j] != 'morph':
                        break

            if verb_idx is None:
                continue

            # 종속절 범위 결정:
            # EC(i)가 직접 children을 가지면(=EC가 어절 root) EC의 subtree 사용
            # 그렇지 않으면 동사 어간(verb_idx)의 subtree 사용
            if self.children[i]:
                subtree_root = i
            else:
                subtree_root = verb_idx
            clause_nodes = self._get_subtree(subtree_root)
            clause_nodes.add(subtree_root)
            # verb_idx ~ EC 사이 형태소들의 subtree도 모두 흡수
            # (SuPar가 어절 내부를 flat하게 분석해 동사 어간이 아닌 EC가 head를 받는 경우 대응)
            for j in range(verb_idx, i + 1):
                clause_nodes.update(self._get_subtree(j))
                clause_nodes.add(j)
            # EC 이후 인덱스 제거
            clause_nodes = {n for n in clause_nodes if n <= i}
            # 이미 확정된 다른 SubC 범위는 이 SubC에서 제외 — 연쇄 EC가 subtree를 공유하는 경우 분리
            for prev_s, prev_e in self.subordinate_ranges:
                clause_nodes -= set(range(prev_s, prev_e + 1))

            # JX/JKS 처리: 종속절 내부 성분이면 어절 전체를 포함, 주절 성분이면 제거
            # clause_nodes 밖의 JX/JKS도 검사 — head가 clause_nodes 안이면 종속절 성분
            topic_subj_nodes: set = set()
            kept_eojeol_nodes: set = set()
            for n in range(i + 1):
                if not self._has_any_tag(n, ('JX', 'JKS')):
                    continue
                head_n = self.arcs[n] - 1 if self.arcs[n] > 0 else -1
                eojeol_s = self._expand_eojeol_left(n)
                # 주어 어절이 NPS(명사 나열)의 일부이면 NPS 전체를 종속절 주어로 포함
                eojeol_s = self._extend_start_with_nps(eojeol_s)
                # 주어 어절 앞에 관형절(AdjP/EmC_Adj)이 인접하면 그 관형절까지 포함
                eojeol_s = self._extend_start_with_adjp(eojeol_s)
                if head_n in clause_nodes:
                    # head가 clause_nodes 내부 → 종속절 성분이므로 어절 전체 추가
                    kept_eojeol_nodes.update(range(eojeol_s, n + 1))
                elif n in clause_nodes:
                    # head가 clause_nodes 밖이고 현재 clause_nodes에 있으면 주절 성분 → 제거
                    topic_subj_nodes.update(range(eojeol_s, n + 1))
            clause_nodes -= topic_subj_nodes
            clause_nodes.update(kept_eojeol_nodes)

            if clause_nodes:
                s, e = min(clause_nodes), i
                # 단일 어절(EC만 있는 경우) SubC는 제외
                non_ec_nodes = [n for n in clause_nodes if not self._has_tag(n, 'EC')]
                if not non_ec_nodes:
                    continue
                self.subordinate_ranges.append((s, e))
                self.spans.append((SpanLabels.SUBC, s, e))

    def _extract_main_clause(self, root_idx: int):
        """주절 추출"""
        # Sentence 전체
        self.spans.append((SpanLabels.SENT, 0, self.N - 1))

        # Main Clause (종속절이 있을 때만)
        if not self.subordinate_ranges:
            return

        # 종속절 끝 이후부터 문장 끝까지를 주절로 잡음
        sub_end = max(e for _, e in self.subordinate_ranges)
        main_start = sub_end + 1

        # 문장부호 제외하고 내용 있는 노드
        content_nodes = [
            i for i in range(main_start, self.N)
            if not self._has_any_tag(i, ('SF', 'SP', 'SE', 'SS'))
        ]

        if content_nodes:
            self.spans.append((SpanLabels.MAINC, min(content_nodes), max(content_nodes)))

    # ========================================================================
    # Phrase 추출
    # ========================================================================

    def _extract_phrases(self):
        """구문 추출"""
        self._extract_verb_phrase()
        # AdjP/EmC_Adj는 extract()에서 SubC 추출 전에 이미 선행 추출됨 (_extract_adjectival)
        # NPS는 extract()에서 SubC 추출 전에 이미 선행 추출됨 (_extract_noun_phrase_sequence)
        self._extract_noun_phrase()             # NP: NPS 범위 내부 중복 제외 후 추가
        self._extract_emc_noun()                # EmC_N: 명사절 내포절 (ETM+NNB/ETN) — NPS 제외 위해 NPS 이후
        self._extract_emc_adv()                 # EmC_Adv: 부사절 내포절 (EC+VP_AJT+논항)
        self._extract_adverbial()               # AdvP: NPS/NP 먼저 확인 후 JKB 확장
        self._extract_topic()                   # TP/SP/OP/CP: NP 위에 조사 붙인 범위
        self._extract_subject()
        self._extract_object()
        self._extract_complement()

    def _extract_topic(self):
        """
        Topic Phrase: N* + JX (은/는)
        N*가 수식을 받는다면 거기까지 모두 포함
        """
        # TP는 보조사 '은/는'(받침 유무에 따른 이형태 포함)만 대상으로 한다.
        # '이나/도/만/까지' 등 다른 보조사는 JX 태그를 공유하지만 TP가 아님.
        _TOPIC_JX = {'은', '는', 'ㄴ', 'ᆫ'}

        for i in range(self.N):
            if not self._is_josa(i, 'JX'):
                continue
            if self.tokens[i] not in _TOPIC_JX:
                continue

            # 바로 앞 형태소가 어미(EC/EF/ETN 등)면 체언+보조사가 아니라
            # 구어체 문말 종결 보조사('해요체'의 요 등) → TP 아님
            if i > 0 and self._has_any_tag(i - 1, ('EC', 'EF', 'ETN')):
                continue

            # 명사구 확장
            s, e = self._expand_np_with_modifier(i)
            # JX 앞에 NPS(명사 나열)가 있으면 NPS 시작까지 확장 (SP/OP/CP와 동일 정책)
            s = self._extend_start_with_nps(s)
            self.spans.append((SpanLabels.TP, s, e))

    def _extract_subject(self):
        """
        Subject Phrase: N* + JKS
        N*가 수식을 받는다면 거기까지 모두 포함
        Subordinate 여부 판단
        """
        for i in range(self.N):
            # JKS 조사 또는 NP_SBJ 관계. 단, VCP(지정사 '이다')는 SuPar가 rel을
            # NP_SBJ로 오예측하는 경우가 있음("사람입니다"의 '이') — 품사가 실제
            # JKS가 아니라 지정사라면 rel만으로 주어로 인정하지 않는다.
            is_subject = self._is_josa(i, 'JKS') or (
                self.rels[i] == 'NP_SBJ' and not self._has_tag(i, 'VCP')
            )
            if not is_subject:
                continue

            # JKS 위치 찾기
            jks_idx = i if self._is_josa(i, 'JKS') else self._find_josa_right(i, 'JKS')
            if jks_idx is None:
                jks_idx = i

            # 같은 어절 내에 JX(보조사/topic marker)가 있으면 TP로 이미 처리되므로 SP 생략
            eojeol_start = self._expand_eojeol_left(jks_idx)
            eojeol_end = jks_idx
            has_topic_marker = any(self._is_josa(k, 'JX') for k in range(eojeol_start, eojeol_end + 1))
            if has_topic_marker:
                continue

            # 명사구 확장
            s, e = self._expand_np_with_modifier(jks_idx)
            s = self._extend_start_with_nps(s)
            self.spans.append((SpanLabels.SP, s, e))

    def _extract_object(self):
        """
        Object Phrase: N* + JKO
        N*가 수식을 받는다면 거기까지 모두 포함
        """
        for i in range(self.N):
            # JKO 조사 또는 NP_OBJ 관계
            is_object = self._is_josa(i, 'JKO') or self.rels[i] == 'NP_OBJ'
            if not is_object:
                continue

            # JKO 위치 찾기
            jko_idx = i if self._is_josa(i, 'JKO') else self._find_josa_right(i, 'JKO')
            if jko_idx is None:
                jko_idx = i

            # 명사구 확장
            s, e = self._expand_np_with_modifier(jko_idx)
            s = self._extend_start_with_nps(s)
            self.spans.append((SpanLabels.OP, s, e))

    def _extract_complement(self):
        """
        Complement Phrase: N* + JKC
        N*가 수식을 받는다면 거기까지 모두 포함

        되다/아니다 앞의 보어
        """
        for i in range(self.N):
            # JKC 조사 또는 NP_CMP 관계
            is_comp = self._is_josa(i, 'JKC') or self.rels[i] == 'NP_CMP'
            if not is_comp:
                continue

            # JKC 위치 찾기
            jkc_idx = i if self._is_josa(i, 'JKC') else self._find_josa_right(i, 'JKC')
            if jkc_idx is None:
                jkc_idx = i

            # 명사구 확장
            s, e = self._expand_np_with_modifier(jkc_idx)
            s = self._extend_start_with_nps(s)
            self.spans.append((SpanLabels.CP, s, e))

    # 부사형 전성어미 (-게, -이, -히, -도록 등): EC이지만 AdvP를 구성
    _ADVERBIAL_EC = {'게', '이', '히', '도록', '듯이', '듯'}

    def _scan_aux_vp_after_jkb(self):
        """JKB(으로/에 등) 뒤에 동사+EC가 바로 이어지는 패턴
        ('으로 인하여/말미암아', '에 대하여/관하여' 등 부사어 동사구)을 미리 찾아 둔다.
        SuPar는 이런 어절 내부에서 JKB와 동사 어간이 어절 대표인 EC를 공통 head로
        가리키는 형제 구조로 분석하는 경우가 있으므로, head 동일성 또는 JKB→동사
        직접 연결 두 패턴을 모두 인식한다.
        VP 추출에서 제외하고 AdvP에 흡수시키기 위해 _extract_phrases 이전에 호출한다.
        """
        for i in range(self.N):
            if not (self._is_josa(i, 'JKB') or self.rels[i] == 'NP_AJT'):
                continue
            jkb_idx = i if self._is_josa(i, 'JKB') else self._find_josa_right(i, 'JKB')
            if jkb_idx is None:
                continue
            jkb_head = self.arcs[jkb_idx] - 1 if self.arcs[jkb_idx] > 0 else -1

            k = jkb_idx + 1
            if not (k < self.N and self._is_verbal(k)):
                continue
            verb_head = self.arcs[k] - 1 if self.arcs[k] > 0 else -1
            # JKB가 동사를 직접 head로 가리키거나, JKB와 동사가 같은 head(어절 대표 EC)를 공유
            if not (verb_head == jkb_idx or (jkb_head >= 0 and jkb_head == verb_head)):
                continue

            # 동사 어간 뒤 어미까지 확장: 일반적인 형태소 사슬(rel=morph, head=이전 형태소)이거나
            # '인하/말미암' 류처럼 어간이 거꾸로 어미를 head로 가리키는 경우(head=다음 형태소) 모두 인정
            m = k
            while m + 1 < self.N and (
                    (self.rels[m + 1] == 'morph' and self.arcs[m + 1] - 1 <= m)
                    or self.arcs[m] - 1 == m + 1):
                m += 1
            # EC 바로 뒤에 보조용언(VX 또는 지/있/없 등)이 오면 이는 부사어 동사구가 아니라
            # '본용언+보조용언'(예: '증후군에 시달리고 있다')이므로 VP가 담당한다 → 제외하지 않음
            if (m + 1 < self.N
                    and self.rels[m + 1] == 'morph'
                    and (self._has_tag(m + 1, 'VX')
                         or self.tokens[m + 1] in self._AUX_VERB_AFTER_EC)):
                continue
            if self._has_tag(m, 'EC'):
                self._aux_vp_after_jkb.add((k, m))

    def _extract_adverbial(self):
        """
        Adverbial Phrase: N* + JKB (부사격조사), 부사(MAG), 또는 부사형 어미(-게 등)
        """
        for i in range(self.N):
            # 1. JKB (부사격조사)
            if self._is_josa(i, 'JKB') or self.rels[i] == 'NP_AJT':
                jkb_idx = i if self._is_josa(i, 'JKB') else self._find_josa_right(i, 'JKB')
                if jkb_idx is not None:
                    # JKB 바로 뒤에 JX가 오면 복합 보조사(처음엔=처음+에+는) → TP의 일부이므로 AdvP 제외
                    if jkb_idx + 1 < self.N and self._has_tag(jkb_idx + 1, 'JX'):
                        continue
                    s, e = self._expand_np_with_modifier(jkb_idx)
                    # JKB 앞에 이미 추출된 NPS 스팬이 있으면 NPS 전체를 AdvP 시작으로 확장
                    # (예: '끈이나 줄 따위로' — NPS[끈이나 줄 따위]가 있으면 s를 NPS 시작으로)
                    s = self._extend_start_with_nps(s)
                    # 사전 스캔한 JKB+동사+EC 부사어 동사구('인하여' 등)가 있으면 AdvP에 포함
                    aux_match = [(k, m) for k, m in self._aux_vp_after_jkb if k == jkb_idx + 1]
                    if aux_match:
                        e = max(e, max(m for _, m in aux_match))
                    self.spans.append((SpanLabels.ADVP, s, e))
                continue

            # 2. 부사형 전성어미 (EC + VP_AJT): -게, -이, -히, -도록 등
            if self._has_tag(i, 'EC') and self.rels[i] == 'VP_AJT' and self.tokens[i] in self._ADVERBIAL_EC:
                # 어절 시작부터 EC까지를 AdvP로
                s = self._expand_eojeol_left(i)
                self.spans.append((SpanLabels.ADVP, s, i))
                continue

            # 3. 부사 (MAG) 또는 AP rel — 단일 형태소는 표시 안 함, EC 태그도 제외
            # (단일 MAG/AP는 글로싱으로 표시되므로 Phrase 바 불필요)
            if (self._has_tag(i, 'MAG') or self.rels[i] == 'AP') and not self._has_tag(i, 'EC'):
                s = self._expand_eojeol_left(i)
                if s == i:  # 단일 형태소 → 표시 안 함
                    continue
                self.spans.append((SpanLabels.ADVP, s, i))

    # "-아/어 + V" 보조용언 연결에서 본용언처럼 VA/VV로 분석되는 보조용언 토큰
    # (지다/있다/없다/버리다/말다 등): 729줄의 VX/XSV 화이트리스트로는 못 잡는 경우 보강
    _AUX_VERB_AFTER_EC = {'지', '있', '없', '버리', '말', '주', '두', '놓', '보'}

    # 'V+ETM+NNB+이다' 패턴에서 명사절(EmC_N) 핵심이 되는 의존명사 — VP가 흡수하면 안 됨
    _NOMINAL_NNB = {'것', '수', '바', '지', '데', '뿐', '터'}

    def _extract_verb_phrase(self):
        """
        Verb Phrase:
        - EF를 포함하는 경우 → Verb Phrase
        - EC를 포함하는 경우 → Verb Phrase (Subordinate)
        - VX가 붙어있는 경우 앞에 있는 V까지 묶음
        - ETM으로 끝나면 → Adjectival Phrase (VP 아님!)
        """
        processed = set()

        for i in range(self.N):
            if i in processed:
                continue

            # 동사/형용사 찾기
            if not self._is_verbal(i):
                continue

            # JKB+동사+EC 부사어 동사구('인하여' 등)는 AdvP가 흡수하므로 VP 제외
            if any(i == k for k, _ in self._aux_vp_after_jkb):
                continue

            # VP 확장: 본용언 + 어미 + 보조용언(VX)
            vp_start = i
            vp_end = i
            has_ef = False
            has_etm = False

            # 오른쪽으로 확장 (같은 어절 내 어미/보조용언 + 대등/보조 연결 허용)
            after_coord_ec = False  # 대등 연결 EC 직후 동사 허용 플래그
            coord_ec_pos = -1       # 가장 최근 대등 EC 위치 (ETM 롤백용)
            for j in range(i, self.N):
                tag = self.xpos[j]
                rel = self.rels[j]

                # 동사/형용사/보조용언
                if self._is_verbal(j):
                    if j == i:
                        vp_end = j
                        after_coord_ec = False
                        coord_ec_pos = -1
                        continue
                    elif after_coord_ec and rel == 'morph':
                        # 대등 연결 EC 직후 동사 → 어절 경계 무시하고 포함
                        # 단, NNG+XSV 합성 동사(공부하다·존중하다 등) 허용
                        vp_end = j
                        after_coord_ec = False
                        # coord_ec_pos는 유지 — 이 어절이 ETM으로 끝날 경우 롤백 필요
                        continue
                    elif rel == 'morph' and vp_start <= self.arcs[j] - 1 <= j:
                        # head가 현재 어절 내에 있음 → 같은 어절 내 형태소
                        vp_end = j
                        continue
                    elif rel == 'morph' and self._has_any_tag(j, ('VX', 'XSV')):
                        # 보조용언(빠져 있다 등): VX는 어절 밖 head라도 허용
                        vp_end = j
                        continue
                    elif (rel == 'morph'
                            and self.tokens[j] in self._AUX_VERB_AFTER_EC
                            and self.tokens[j - 1] in ('아', '어')):
                        # 보조용언(없어지다 등)이 VA/VV로 분석된 경우: 어절 밖 head라도 허용
                        vp_end = j
                        continue
                    else:
                        break

                # EF/EP/ETM — 어절 내 어미
                if tag.startswith(('EF', 'EP', 'ETM')):
                    if tag.startswith('ETM') and coord_ec_pos >= 0:
                        # 대등 EC 이후 어절이 ETM으로 끝남 → 그 어절은 VP 밖(AdjP/EmC_N 담당)
                        # vp_end를 대등 EC 위치로 되돌리고 VP 확정
                        vp_end = coord_ec_pos
                        break
                    vp_end = j
                    after_coord_ec = False
                    if tag.startswith('EF'):
                        has_ef = True
                        # EF 직후 구어체 청자높임 보조사 '요'(JX, rel=morph)가 오면 VP에 포함
                        if (j + 1 < self.N
                                and self._has_tag(j + 1, 'JX')
                                and self.tokens[j + 1] == '요'
                                and self.rels[j + 1] == 'morph'):
                            vp_end = j + 1
                            break
                    elif tag.startswith('ETM'):
                        has_etm = True
                    continue

                # EC — 연결어미
                if tag.startswith('EC'):
                    vp_end = j
                    after_coord_ec = False
                    # 보조용언 연결 EC(어/아 + VX): 다음 형태소가 VX면 계속 확장
                    if (j + 1 < self.N
                            and self.rels[j + 1] == 'morph'
                            and self._has_any_tag(j + 1, ('VX', 'XSV'))):
                        continue
                    # 보조용언 연결 EC(어/아 + 지다/있다/없다 등): VA/VV로 분석되지만
                    # 보조용언으로 쓰이는 토큰 화이트리스트로 계속 확장
                    if (self.tokens[j] in ('아', '어')
                            and j + 1 < self.N
                            and self.rels[j + 1] == 'morph'
                            and self.tokens[j + 1] in self._AUX_VERB_AFTER_EC):
                        continue
                    # 대등 연결 EC(고/며/거나 등): 다음 어절이 용언이면 계속 확장 (부사절 VP_AJT로 끝나면 중단)
                    _COORD_EC = {
                        '고', '며', '으며', '거나', '든지', '든가', '으나', '나',
                    }
                    if (self.tokens[j] in _COORD_EC
                            and j + 1 < self.N
                            and self.rels[j + 1] == 'morph'
                            and self._is_verbal(j + 1)):
                        # 다음 어절의 마지막 어미를 찾아서 rel 확인
                        # rel=morph인 EC는 어절 내부이므로 통과; rel이 다른 EC가 어절 끝
                        k = j + 1
                        while k + 1 < self.N:
                            nxt_rel = self.rels[k + 1]
                            nxt_tag = self.xpos[k + 1]
                            if (nxt_rel == 'morph'
                                    or self._is_verbal(k + 1)
                                    or nxt_tag.startswith(('EF', 'EP', 'ETM', 'X'))):
                                k += 1
                            elif nxt_tag.startswith('EC') and nxt_rel == 'morph':
                                # 어절 내부 EC(보조용언 연결 어/아 등) → 통과
                                k += 1
                            else:
                                break
                        # k+1이 VP_AJT/VP_CMP rel을 가진 EC이면 k+1이 어절 끝 어미
                        if k + 1 < self.N and self.xpos[k + 1].startswith('EC'):
                            k = k + 1
                        if self.rels[k] in ('VP_AJT', 'VP_CMP'):
                            break
                        after_coord_ec = True
                        coord_ec_pos = j  # ETM 롤백 대비 EC 위치 기록
                        continue
                    # 구어체 문말 종결 보조사(는데요/어서요 등): EC 뒤에 JX가 바로 오고
                    # 그 뒤로 후속 용언/절 없이 문장이 끝나면 JX까지 VP에 포함
                    if (j + 1 < self.N and self._has_tag(j + 1, 'JX')):
                        has_following_clause = False
                        for k in range(j + 2, self.N):
                            if self._is_verbal(k):
                                has_following_clause = True
                                break
                            if self._has_any_tag(k, ('SF', 'SP', 'SE', 'SS')):
                                continue
                            has_following_clause = True
                            break
                        if not has_following_clause:
                            vp_end = j + 1
                            continue
                    break  # 그 외 EC → 어절 경계로 보고 중단

                # morph 관계 + 동사/어미류만
                if rel == 'morph' and (self._is_verbal(j) or tag.startswith('E') or tag.startswith('X')):
                    vp_end = j
                    continue

                # 접미사(XSV/XSA)
                if tag.startswith(('XSV', 'XSA')):
                    vp_end = j
                    continue

                # 명사/조사 등 만나면 중단
                break

            # ETM으로 끝나면 VP가 아님 (Adjectival)
            if has_etm:
                processed.update(range(vp_start, vp_end + 1))
                continue

            # 마지막 어미가 VP_AJT 관계이면 AdvP 형성 → VP 아님 (이렇게, 명확하게 등)
            if self.rels[vp_end] == 'VP_AJT':
                processed.update(range(vp_start, vp_end + 1))
                continue

            # 마지막 EC가 인용 신호(라고 등)이면 VP 아님 (QuoteC_Ind가 담당)
            # 단, 'NNB(것/거)+VCP+EC' 패턴('것이라고')은 예외로 VP 유지
            is_nnb_vcp_ec = (
                i >= 2
                and self._has_tag(i, 'VCP')
                and self._has_tag(i - 1, 'NNB')
                and self.tokens[i - 1] in ('것', '거')
            )
            if (self.xpos[vp_end].startswith('EC')
                    and self._is_quote_trigger(vp_end)
                    and not is_nnb_vcp_ec):
                processed.update(range(vp_start, vp_end + 1))
                continue

            # 왼쪽으로 확장 (VX/XSV 앞의 V 및 NNG+XSV 합성 동사 어간)
            for j in range(i - 1, -1, -1):
                if self._is_verbal(j):
                    if self.rels[i] == 'morph' or self.rels[i] == 'VP':
                        vp_start = j
                    else:
                        break
                elif (self.rels[j] == 'morph'
                        and self._has_any_tag(j, ('NNG', 'NNP', 'NNB', 'XSN'))
                        and self.tokens[j] not in self._NOMINAL_NNB):
                    # 말하다·공부하다 등 NNG+XSV 패턴에서 NNG를 VP에 포함
                    # 단, 것/수/바/지/데/뿐/터 등 의존명사+이다는 EmC_N 소속이므로 제외
                    vp_start = j
                else:
                    break

            # '것 같다' 패턴: 동사 왼쪽에 NNB(것) + ETM이 있으면 ETM 어절까지 포함
            if vp_start > 0:
                left = vp_start - 1
                if (self._has_tag(left, 'NNB')
                        and self.rels[left] == 'NP'
                        and left > 0
                        and self._has_tag(left - 1, 'ETM')):
                    # ETM 어절 시작까지 역추적
                    etm_idx = left - 1
                    etm_start = etm_idx
                    for k in range(etm_idx - 1, -1, -1):
                        if self.rels[k] == 'morph' and self.arcs[k] - 1 > k:
                            etm_start = k
                        else:
                            break
                    vp_start = etm_start
                # '올 것이라고/될 거라고' 패턴: NNB(것/거)+ETM이 rel=morph로 이어지고 EC로 끝나면
                # ETM 어절까지 하나의 VP로 (것을 EmC_N으로 떼지 않음)
                elif (not has_ef
                        and self._has_tag(vp_start, 'VCP')
                        and self._has_tag(left, 'NNB')
                        and self.tokens[left] in ('것', '거')
                        and self.rels[left] == 'morph'
                        and left > 0
                        and self._has_tag(left - 1, 'ETM')):
                    etm_idx = left - 1
                    etm_start = etm_idx
                    for k in range(etm_idx - 1, -1, -1):
                        if self._is_verbal(k):
                            etm_start = k
                            break
                        if self.rels[k] == 'morph' or self._has_any_tag(k, ('E', 'X')):
                            etm_start = k
                            continue
                        break
                    vp_start = etm_start

            # 라벨 결정
            self.spans.append((SpanLabels.VP, vp_start, vp_end))
            processed.update(range(vp_start, vp_end + 1))

    def _extract_adjectival(self):
        """
        Adjectival Phrase:
        - VP_MOD relation을 활용한 관형절 추출
        - JKG가 붙어 있다면 Adjectival Phrase (possessive)
        """
        processed = set()

        # 1. 관형형 전성어미(ETM) 기반 관형절 추출
        #    관형형 어미 ㄴ/은/는/ㄹ 등은 명사를 수식하는 관형절의 핵이다. ETM은 정의상 항상
        #    관형형이지만 SuPar는 관형절 위치에 따라 rel을 VP_MOD/VP/NP 등으로 제각각 붙인다
        #    (연쇄 시 마지막만 VP_MOD, '친구가 준 책'의 '준'은 NP). 따라서 ETM은 rel로 거르지
        #    않고 아래 guard(용언 head/명사화 의존명사 등)로 비관형 용법만 배제한다.
        #    XSV/XSA(파생접미사)는 관형절 관계(VP_MOD/VP)일 때만 취한다.
        for i in range(self.N):
            is_etm = self._has_tag(i, 'ETM')
            if not (is_etm or self._has_tag(i, 'XSV') or self._has_tag(i, 'XSA')):
                continue

            # "-고 계신" 등 보조용언 뒤 ETM을 SuPar가 NP/VP로 오분석하는 경우 — 앞이 VX면 관형절로 인정
            is_aux_np_misparse = (
                i > 0
                and self.rels[i] in ('NP', 'VP')
                and self._has_tag(i - 1, 'VX')
            )
            if not is_etm and self.rels[i] not in ('VP_MOD', 'VP') and not is_aux_np_misparse:
                continue

            # VP_MOD의 head를 찾음 (ETM이 수식하는 명사/의존명사의 head)
            etm_head_idx = self.arcs[i] - 1 if i < len(self.arcs) and self.arcs[i] > 0 else -1

            # ETM head가 용언이면 VP 내부 수식 → AdjP 아님
            if etm_head_idx >= 0 and self._is_verbal(etm_head_idx):
                continue

            # ETM 바로 뒤가 명사화 의존명사(것/수/바/지/데/뿐/터)면 이는 관형절이 아니라
            # 명사절 내포절(EmC_N)의 핵이다 → 관형절 분기를 통째로 건너뛰고 _extract_emc_noun에
            # 맡긴다. (SuPar 파스에서 ETM의 head는 '것'이 아니라 뒤의 조사/지정사를 가리키는
            # 경우가 많아 head가 아니라 바로 다음 토큰으로 판정한다. 정답의 EmC_Adj는 모두 일반
            # 명사(땅/상황/사람/때 등)를 수식하며 '것/수' 등 명사화 의존명사 수식 예는 없다.)
            if (i + 1 < self.N
                    and self._has_tag(i + 1, 'NNB')
                    and self.tokens[i + 1] in self._NOMINAL_NNB):
                continue

            # ETM head가 의존명사(NNB)이고 그 NNB의 rel이 NP(주절 VP 소속)이면
            # '많은 것 같다' 같은 추측 구문 내부 → AdjP 아님
            if (etm_head_idx >= 0
                    and self._has_tag(etm_head_idx, 'NNB')
                    and self.rels[etm_head_idx] == 'NP'):
                continue

            # ETM 뒤에 NNB(것/거)+VCP+EC가 이어지면 '올 것이라고' 구문이라 VP가 담당 → AdjP 아님
            if (i + 3 < self.N
                    and self._has_tag(i + 1, 'NNB')
                    and self.tokens[i + 1] in ('것', '거')
                    and self._has_tag(i + 2, 'VCP')
                    and self._has_tag(i + 3, 'EC')):
                continue

            # ETM 바로 앞 어절의 핵어 동사만 찾음 (전체 verbal 수집 금지)
            verb_idx = None
            for j in range(i - 1, -1, -1):
                if self._is_verbal(j):
                    verb_idx = j
                    break
                if self.rels[j] not in ('morph',) and not self._has_any_tag(j, ('E', 'X')):
                    break

            if verb_idx is None:
                continue

            # 보조용언 구성(배달해 드릴)처럼 동일 어절 내 선행 형태소(본용언)가
            # rel=morph로 verb_idx를 head로 가리키면 그 어절 시작까지 확장
            verb_idx = self._expand_eojeol_left(verb_idx)

            # "신고 있는/계신"처럼 본용언+EC+보조용언이 별도 어절로 분석되면(EC rel=VP) 본용언까지 확장
            if (verb_idx >= 2
                    and self._has_tag(verb_idx, 'VX')
                    and self._has_any_tag(verb_idx - 1, ('EC',))
                    and self.rels[verb_idx - 1] == 'VP'):
                main_verb_idx = self._expand_eojeol_left(verb_idx - 2)
                if self._is_verbal(main_verb_idx):
                    verb_idx = main_verb_idx

            # 서술어 어절(관형절의 핵) 범위: verb_idx 어절 시작 ~ i(ETM)
            pred_start = self._expand_eojeol_left(verb_idx)
            clause_end = i

            # (1) 서술어만의 관형구는 항상 AdjP로 추출 (예: 아닌/따뜻한/좋아하는)
            key = (pred_start, clause_end)
            if key not in processed:
                processed.add(key)
                self.spans.append((SpanLabels.ADJP, pred_start, clause_end))

            # (2) 관형절 내부 논항(주어/목적어/보어/부사어) 수집 — 서술어 어절 형태소의
            #     '논항 rel 직속 자식'과 그 subtree만 취한다. 이렇게 하면 SuPar 연쇄 파스에서
            #     서술어 어미에 형제로 매달린 별개 관형절(rel=VP/VP_MOD)은 논항이 아니므로
            #     자연히 배제되어(예: '따뜻한'에 '얼음이 아닌'이 섞이지 않음) 범위 오염을 막는다.
            #     논항이 있으면 논항+서술어 전체를 EmC_Adj(관형절 내포절)로 AdjP와 중첩 추출한다.
            #     명사형 전성어미(기/ㅁ)로 명사화된 절 논항은 VP_OBJ/VP_SBJ 등으로 붙으므로
            #     함께 인정한다(예: '여행하기 좋은'의 목적어 '여행하기'는 VP_SBJ).
            _ARG_RELS = {'NP_SBJ', 'NP_OBJ', 'NP_CMP', 'NP_AJT', 'AP', 'NP_CNJ',
                         'VP_OBJ', 'VP_SBJ', 'VP_CMP', 'VP_AJT'}
            arg_nodes: set = set()
            for node in range(pred_start, clause_end + 1):
                for c in self.children[node]:
                    if c >= pred_start or self.rels[c] not in _ARG_RELS:
                        continue
                    # 관용구 '-ㄹ 수 있다/없다'의 '수'처럼 조사 없이 붙은 의존명사(NNB)는
                    # 실질 논항이 아니라 문법화된 구성이므로 EmC_Adj 트리거에서 제외한다.
                    if self._has_tag(c, 'NNB'):
                        continue
                    arg_nodes |= self._get_subtree(c)
            if arg_nodes:
                clause_start = min(min(arg_nodes), pred_start)
                if clause_start < pred_start:
                    ekey = (clause_start, clause_end)
                    if ekey not in processed:
                        processed.add(ekey)
                        self.spans.append((SpanLabels.EMC_ADJ, clause_start, clause_end))

        # 2. NP_MOD("의") - 명사구 + "의" 전체를 Adjectival Phrase로
        # "의"는 JKG가 아니라 JX 태그로 나타나며, 관계는 NP_MOD
        for i in range(self.N):
            # "의" 형태소이고 NP_MOD 관계인 경우
            if self.tokens[i] != '의':
                continue
            if i >= len(self.rels) or self.rels[i] != 'NP_MOD':
                continue

            # 왼쪽으로 명사구 확장
            clause_start = i
            for j in range(i - 1, -1, -1):
                # 명사, 관형사, 수사 등 계속 포함
                if self._has_tag(j, 'NN') or self._has_tag(j, 'NP') or self._has_tag(j, 'NR') or self._has_tag(j, 'MM'):
                    clause_start = j
                # morph 관계면 계속 (복합명사, 파생어 등)
                elif j < len(self.rels) and self.rels[j] == 'morph':
                    continue
                # 다른 조사나 구분 만나면 중단
                else:
                    break

            key = (clause_start, i)
            if key not in processed:
                processed.add(key)
                self.spans.append((SpanLabels.ADJP, clause_start, i))

        # 3. MM("다른", "같은" 등) 관형사
        for i in range(self.N):
            if not self._has_tag(i, 'MM'):
                continue

            # MM 왼쪽으로 확장 (부사 등 포함)
            clause_start = i
            for j in range(i - 1, -1, -1):
                if self._has_tag(j, 'MAG'):  # 부사
                    clause_start = j
                elif j < len(self.rels) and self.rels[j] == 'morph':
                    continue
                else:
                    break

            # 단일 형태소 MM은 AdjP로 표시하지 않음
            if clause_start == i:
                continue

            key = (clause_start, i)
            if key not in processed:
                processed.add(key)
                self.spans.append((SpanLabels.ADJP, clause_start, i))

        # 4. 부정문 ETM: "지 않은", "지 못한" 등
        for i in range(self.N):
            if self.tokens[i] != '지':
                continue
            if i + 2 >= self.N:
                continue

            # "지 않/못 + ETM" 패턴
            if self.tokens[i + 1] in ['않', '못']:
                if self._has_tag(i + 2, 'ETM') or self._has_tag(i + 2, 'XSV'):
                    # 왼쪽으로 동사/형용사 찾기
                    verb_idx = None
                    for j in range(i - 1, -1, -1):
                        if self._is_verbal(j):
                            verb_idx = j
                            break
                        if j < len(self.rels) and self.rels[j] != 'morph':
                            break

                    if verb_idx is not None:
                        # Subtree 추출
                        clause_nodes = self._get_subtree(verb_idx)
                        clause_nodes = {n for n in clause_nodes if n <= i + 2}
                        if clause_nodes:
                            clause_start = min(clause_nodes)
                            clause_end = i + 2
                            key = (clause_start, clause_end)
                            if key not in processed:
                                processed.add(key)
                                self.spans.append((SpanLabels.ADJP, clause_start, clause_end))

        # 5. 파생 형용사 ETM: "구체적이ㄴ" (XR + VCP + ETM)
        for i in range(self.N):
            if not self._has_tag(i, 'VCP'):  # 긍정 지정사 "이다"
                continue
            if i + 1 >= self.N:
                continue
            if not (self._has_tag(i + 1, 'ETM') or self._has_tag(i + 1, 'XSV') or self._has_tag(i + 1, 'XSA')):
                continue

            # 왼쪽에 어근(XR) 또는 명사 찾기
            clause_start = i
            for j in range(i - 1, -1, -1):
                if self._has_tag(j, 'XR') or self._has_tag(j, 'NN'):
                    clause_start = j
                elif j < len(self.rels) and self.rels[j] == 'morph':
                    continue
                else:
                    break

            clause_end = i + 1
            key = (clause_start, clause_end)
            if key not in processed:
                processed.add(key)
                self.spans.append((SpanLabels.ADJP, clause_start, clause_end))

        # 6. 보조용언 ETM: "할 수 없는", "하기 좋은" 등
        for i in range(self.N):
            # "수/것" + 보조용언 + ETM 패턴
            if self.tokens[i] in ['수', '것']:
                if i + 2 >= self.N:
                    continue
                # 보조용언 확인
                if self._is_verbal(i + 1):
                    if self._has_tag(i + 2, 'ETM') or self._has_tag(i + 2, 'XSV') or self._has_tag(i + 2, 'XSA'):
                        # 왼쪽으로 본동사 찾기
                        verb_idx = None
                        for j in range(i - 1, -1, -1):
                            if self._is_verbal(j):
                                verb_idx = j
                                break

                        if verb_idx is not None:
                            # Subtree 추출
                            clause_nodes = self._get_subtree(verb_idx)
                            clause_nodes = {n for n in clause_nodes if n <= i + 2}
                            if clause_nodes:
                                clause_start = min(clause_nodes)
                                clause_end = i + 2
                                key = (clause_start, clause_end)
                                if key not in processed:
                                    processed.add(key)
                                    self.spans.append((SpanLabels.ADJP, clause_start, clause_end))

        # 7. XR + XSA 직접 패턴 (인용절 내부 등 VP_MOD 관계 없을 때)
        # 예: "똑똑 하" (XR + XSA)
        for i in range(self.N - 1):
            if not self._has_tag(i, 'XR'):
                continue
            if not self._has_tag(i + 1, 'XSA'):
                continue

            clause_start = i
            clause_end = i + 1
            key = (clause_start, clause_end)
            if key not in processed:
                processed.add(key)
                self.spans.append((SpanLabels.ADJP, clause_start, clause_end))

    def _extract_emc_noun(self):
        """명사절 내포절(EmC_N) 추출.
        트리거:
          A) ETM + NNB('것'/'수'/'바'/'지'/'데'/'뿐'/'터') 패턴
          B) ETN(명사형 전성 어미) 직접 종결 패턴
        내부 논항이 없으면 EmC_N 대신 단순 NP로 처리(AdjP+NNB 구성).
        """
        processed: set = set()

        for i in range(self.N):
            is_etn = self._has_tag(i, 'ETN')
            # ETM + NNB 패턴: i가 NNB이고 바로 앞이 ETM
            is_nnb_after_etm = (
                self._has_tag(i, 'NNB')
                and self.tokens[i] in self._NOMINAL_NNB
                and i > 0
                and self._has_tag(i - 1, 'ETM')
            )

            if not (is_etn or is_nnb_after_etm):
                continue

            end_idx = i
            etm_pos = (i - 1) if is_nnb_after_etm else i

            # 절 범위: ETM subtree에서 etm_pos 이하만 취하되, 별도 절/구(SubC/NPS/AdjP/EmC_Adj/AdvP)는 제외
            _EXCL_LABELS = {SpanLabels.NPS, SpanLabels.ADJP, SpanLabels.EMC_ADJ, SpanLabels.ADVP}
            excluded_nodes: set = set()
            for ss, se in self.subordinate_ranges:
                if se < etm_pos:
                    excluded_nodes.update(range(ss, se + 1))
            for lbl, ss, se in self.spans:
                if lbl in _EXCL_LABELS and se < etm_pos:
                    excluded_nodes.update(range(ss, se + 1))

            clause_nodes = self._get_subtree(etm_pos)
            clause_nodes = {n for n in clause_nodes if n <= etm_pos and n not in excluded_nodes}

            # 대등이 아닌 EC(rel=VP/VP_AJT)가 subtree 내에 있으면 해당 EC 포함 이전은 별도 절
            # → 가장 오른쪽 해당 EC+1 이후만 취함
            _NON_COORD_EC_RELS = {'VP', 'VP_AJT'}
            _COORD_EC_TOKENS = {'고', '며', '으며', '거나', '든지', '든가', '으나', '나'}
            for n in sorted(clause_nodes, reverse=True):
                if (n < etm_pos
                        and self._has_tag(n, 'EC')
                        and self.rels[n] in _NON_COORD_EC_RELS
                        and self.tokens[n] not in _COORD_EC_TOKENS):
                    clause_nodes = {k for k in clause_nodes if k > n}
                    break

            # 쉼표(SP)/인용부호(SS) 경계를 넘지 않도록: 가장 오른쪽 구분 부호 이후만 취함
            for n in sorted(clause_nodes):
                if self._has_any_tag(n, ('SP', 'SS')) and n < etm_pos:
                    clause_nodes = {k for k in clause_nodes if k > n}

            # 주절 주어(JX/JKS) 제거 — head가 명사절 핵심(NNB/ETN)이거나 clause_nodes 밖이면 주절 주어
            topic_subj_nodes: set = set()
            for n in sorted(clause_nodes):
                if not self._has_any_tag(n, ('JX', 'JKS')):
                    continue
                head_n = self.arcs[n] - 1 if self.arcs[n] > 0 else -1
                if head_n not in clause_nodes or head_n == etm_pos:
                    eojeol_s = self._expand_eojeol_left(n)
                    topic_subj_nodes.update(k for k in clause_nodes if eojeol_s <= k <= n)
            if topic_subj_nodes:
                remaining = clause_nodes - topic_subj_nodes
                # 제거 후에도 남은 노드 중 topic_subj_nodes 이후(오른쪽)만 유지 — 주어는 항상 절 앞쪽
                if remaining:
                    cutoff = max(topic_subj_nodes)
                    clause_nodes = {k for k in remaining if k > cutoff}

            if not clause_nodes:
                continue
            start_idx = min(clause_nodes)

            key = (start_idx, end_idx)
            if key in processed:
                continue
            processed.add(key)

            # 내부 논항이 있으면 EmC_N, 없으면 단순 NP(수식절 없는 구)
            if self._has_internal_arguments(None, start_idx, etm_pos):
                self.spans.append((SpanLabels.EMC_N, start_idx, end_idx))
                # EmC_N으로 흡수된 'V+ETM(+NNB)' 구조를 _extract_adjectival/_extract_noun_phrase가
                # 이미 AdjP/NP로 따로 내놓았다면 중복이므로 제거 (관형절+의존명사를 두 번 쪼개지 않음)
                self.spans = [
                    (lbl, s, e) for lbl, s, e in self.spans
                    if not (lbl in (SpanLabels.ADJP, SpanLabels.NP)
                            and start_idx <= s and e in (etm_pos, end_idx))
                ]
            else:
                # 논항 없는 'V+ETM+NNB' → NP로 처리 (기존 _extract_adjectival + NP 로직과 일치)
                pass

        # C) 의문형 종결어미로 명사 기능을 하는 절 → EmC_N
        #    종결어미는 문장을 끝내는 역할 외에, 하나의 절을 이루면서 그 절이 명사처럼
        #    문장 성분(주어/목적어/부사어)으로 쓰이게 하는 경우가 있다. 의문형 어미
        #    (느냐/는지/을지/는가/을까 등) 뒤에 격조사가 붙어 논항으로 기능하면 명사절 내포절이다.
        #    (예: '부부가 얼마나 서로 잘 맞춰 사느냐에 달려 있다'의 '부부가 ... 사느냐')
        _INTERROG_END = {'느냐', '으냐', '냐', '는지', 'ᆫ지', 'ㄴ지', '은지', '을지', 'ᆯ지', 'ㄹ지',
                         '는가', 'ᆫ가', 'ㄴ가', '은가', '을까', 'ᆯ까', 'ㄹ까'}
        for i in range(self.N):
            if not self._has_any_tag(i, ('EC', 'EF')):
                continue
            if self.tokens[i] not in _INTERROG_END:
                continue
            # 바로 뒤에 격조사가 붙어야 절이 문장 성분(명사절)으로 기능한다
            mk = i + 1
            if mk >= self.N or not self._has_any_tag(mk, ('JKB', 'JKO', 'JKS', 'JKC', 'JKG')):
                continue
            # 절 범위 = 격조사의 subtree에서 의문형 어미(i) 이하만 (격조사·이후 제외)
            clause_nodes = {n for n in self._get_subtree(mk) if n <= i}
            if not clause_nodes:
                continue
            start_idx = min(clause_nodes)
            key = (start_idx, i)
            if key in processed:
                continue
            processed.add(key)
            if self._has_internal_arguments(None, start_idx, i):
                self.spans.append((SpanLabels.EMC_N, start_idx, i))

    def _extract_emc_adv(self):
        """부사절 내포절(EmC_Adv) 추출.
        트리거: EC이면서 rel=VP_AJT이고 내부 논항이 있는 경우.
        내부 논항이 없으면 단순 AdvP로 남겨 둔다(기존 _extract_adverbial이 처리).
        """
        for i in range(self.N):
            if not self._has_tag(i, 'EC'):
                continue
            if self.rels[i] != 'VP_AJT':
                continue
            # 인용절 내부 제외
            if any(qs <= i <= qe for qs, qe in self.quotation_ranges):
                continue

            # EC 앞 동사 찾기
            verb_idx = None
            for j in range(i - 1, -1, -1):
                if self._is_verbal(j):
                    verb_idx = j
                    break
                if self.rels[j] not in ('morph',) and not self._has_any_tag(j, ('E', 'X')):
                    break
            if verb_idx is None:
                continue
            verb_idx = self._expand_eojeol_left(verb_idx)

            clause_nodes = self._get_subtree(verb_idx)
            clause_nodes = {n for n in clause_nodes if n <= i}
            if not clause_nodes:
                continue
            start_idx = min(clause_nodes)

            # 내부 논항이 있어야 절(EmC_Adv), 없으면 단순 AdvP로 남김
            if self._has_internal_arguments(verb_idx, start_idx, i):
                self.spans.append((SpanLabels.EMC_ADV, start_idx, i))

    def _extract_noun_phrase(self):
        """
        Noun Phrase: 명사까지 묶어서 Noun Phrase로 냄
        - 조사가 붙은 경우: 조사 제외한 명사 부분을 NP로
        - 관형절이 붙은 경우: 관형절 포함해서 NP로
        - VCP(이다) 앞의 명사도 NP로
        - 예: "영어로 하는 수업이" → "영어로 하는 수업" (NP) + "수업이" (SP)
        - 예: "요즘 장마철이라" → "요즘 장마철" (NP)
        """
        processed = set()

        # 조사가 붙은 명사구에서 조사 제외한 부분을 NP로
        for i in range(self.N):
            # 조사(JK*, JX) 또는 VCP(이다/아니다) 찾기
            is_josa = self._has_any_tag(i, ('JKS', 'JKO', 'JKB', 'JKG', 'JKC', 'JX'))
            is_copula = self._has_tag(i, 'VCP')  # 이다 (긍정지정사)
            # 연결조사(과/와/나, rel=NP_CNJ) 앞의 명사(구)도 접속항 NP 후보로 본다.
            # 정답 데이터는 관형어가 붙은 접속항(예: '소외된 계층', '오랜 세월')을 NP로 표기한다.
            # 단순 명사 접속항은 아래 단일 형태소 억제 규칙이, 나열 전체 NP는 has_modifier 필터가 거른다.
            is_conj = self._has_tag(i, 'JC') and self.rels[i] == 'NP_CNJ'

            if not is_josa and not is_copula and not is_conj:
                continue

            # 조사 바로 앞이 명사인지 확인
            if i == 0:
                continue

            # VCP(이) 앞이 NNB(것/거)이고 그 뒤가 EC(EF 아님, 문장이 이어짐)이면
            # '올 것이라고' 같은 시제+인용 구문 전체가 VP로 묶이므로 NP 아님
            if (is_copula
                    and self._has_tag(i - 1, 'NNB')
                    and self.tokens[i - 1] in ('것', '거')
                    and i + 1 < self.N
                    and self._has_tag(i + 1, 'EC')):
                continue

            noun_end = i - 1
            if not self._is_nominal(noun_end) and not self._has_any_tag(noun_end, ('XSN',)):
                # morph로 연결된 명사 찾기
                for j in range(i - 1, -1, -1):
                    if self._is_nominal(j):
                        noun_end = j
                        break
                    if self.rels[j] != 'morph':
                        break

            if noun_end < 0:
                continue

            # noun_end가 여전히 명사류가 아니면(조사·어미 등) NP 불가
            if not self._is_nominal(noun_end) and not self._has_any_tag(noun_end, ('XSN', 'SL', 'SN')):
                continue

            # 명사구 시작점 찾기 (관형절 AdjP 포함). while 기반: 왼쪽 어절로 점프 후 j=np_start-1로
            # 재동기화 (순차 감소는 건너뛴 어절 내부에서 끊김. _expand_np_with_modifier와 동일 정책)
            np_start = noun_end
            j = noun_end - 1
            while j >= 0:
                rel = self.rels[j]

                # NP_MOD (관형격 "의") — 조사 자체이므로 명사류 체크 없이 통과
                if rel == 'NP_MOD':
                    np_start = self._expand_eojeol_left(j)
                    j = np_start - 1
                    continue

                # 관형절(ETM/XSV/XSA로 끝나는 어절) — 이미 추출된 AdjP/EmC_Adj가 있으면 그 시작까지 확장
                # (rel=VP_MOD가 아니어도 ETM이면 처리. DP/NP 분기보다 먼저 둬야 rel=NP인 ETM이 잡힘)
                if self._has_any_tag(j, ('ETM', 'XSV', 'XSA')):
                    adj_match = [
                        (s, e) for lbl, s, e in self.spans
                        if lbl in (SpanLabels.ADJP, SpanLabels.EMC_ADJ) and s <= j <= e
                    ]
                    if adj_match:
                        # 가장 큰(포함 범위 최대) AdjP 사용
                        adj_match.sort(key=lambda x: x[1] - x[0], reverse=True)
                        np_start = adj_match[0][0]
                        break  # 관형절 시작 확정 후 더 이상 왼쪽 탐색 불필요
                    elif rel == 'VP_MOD':
                        # fallback: ETM 어절 morph chain 시작점만 사용
                        np_start = j
                        for k in range(j - 1, -1, -1):
                            if self.rels[k] == 'morph' and self.arcs[k] - 1 > k:
                                np_start = k
                            else:
                                break
                        break  # 관형절 시작 확정 후 더 이상 왼쪽 탐색 불필요
                    # ETM이지만 확장 근거(AdjP/VP_MOD)가 없으면 아래 일반 분기(복합명사 morph)에 위임

                # DP (관형사), NP (복합명사) — 해당 형태소가 명사/관형사류여야 함
                if rel in {'DP', 'NP'}:
                    if not (self._is_nominal(j) or self._has_any_tag(j, ('MM', 'SL', 'SN', 'XPN', 'XSN'))):
                        break
                    # MM(관형사)을 포함하는 AdjP가 이미 있으면 AdjP 시작으로 확장 (예: '서로 다른 선택')
                    if self._has_any_tag(j, ('MM', 'MMA', 'MMD', 'MMN')):
                        adj_match = [
                            (s, e) for lbl, s, e in self.spans
                            if lbl == SpanLabels.ADJP and s <= j <= e
                        ]
                        if adj_match:
                            adj_match.sort(key=lambda x: x[1] - x[0], reverse=True)
                            np_start = adj_match[0][0]
                            j = np_start - 1
                            continue
                    np_start = self._expand_eojeol_left(j)
                    j = np_start - 1
                    continue

                # 관형사(MM): 그 MM을 포함하는 AdjP가 이미 있으면 AdjP 시작으로 확장
                # AdjP가 없으면 MM 자체만 포함.
                if self._has_any_tag(j, ('MM', 'MMA', 'MMD', 'MMN')):
                    adj_match = [
                        (s, e) for lbl, s, e in self.spans
                        if lbl == SpanLabels.ADJP and s <= j <= e
                    ]
                    if adj_match:
                        adj_match.sort(key=lambda x: x[1] - x[0], reverse=True)
                        np_start = adj_match[0][0]
                        j = np_start - 1
                    else:
                        np_start = j
                        j -= 1
                    continue

                # 명사류 연속 — 단, 부사어/수식어로 쓰인 명사(NP_AJT)는 복합명사가 아님
                if self._is_nominal(j) and rel not in {'NP_AJT', 'NP_SBJ', 'NP_OBJ', 'NP_CMP'}:
                    np_start = j
                    j -= 1
                    continue

                # morph 관계
                if rel == 'morph':
                    np_start = j
                    j -= 1
                    continue

                break

            # 중복 체크 및 추가
            key = (np_start, noun_end)
            if key not in processed and np_start <= noun_end:
                processed.add(key)
                # 단일 형태소 NP는 표시 안 함.
                # 단, xpos에 '+' 포함 시 합성어/파생어이므로 표시함
                if np_start == noun_end and '+' not in self.xpos[np_start]:
                    continue
                # NPS(명사 나열) 안의 NP 처리:
                #   기본은 제외(나열 전체·개별 명사는 NPS가 대표). 단, 관형어(MM/관형절/의)를
                #   품은 접속항 NP(예: '소외된 계층', '오랜 세월', '국보이었던 극락전')는
                #   정답 데이터가 별도로 표기하므로 유지한다. 연결조사(NP_CNJ)를 품어 나열
                #   전체를 덮는 NP는 항상 제외.
                in_nps = any(ns <= np_start and noun_end <= ne
                             for lbl, ns, ne in self.spans if lbl == SpanLabels.NPS)
                if in_nps:
                    # 관형어(MM/관형절/의)를 품은 NP만 유지한다. 나열 전체를 관형어가 수식하는
                    # 경우(예: '구체적인 사례나 사건을')는 정답도 NP로 표기하므로 연결조사(NP_CNJ)
                    # 포함 여부와 무관하게 유지하고, 관형어 없는 단순 나열/개별 명사는 제외한다.
                    has_modifier = any(
                        self._has_any_tag(k, ('MM', 'ETM', 'XSV', 'XSA'))
                        or self.rels[k] == 'NP_MOD'
                        for k in range(np_start, noun_end + 1)
                    )
                    if not has_modifier:
                        continue
                self.spans.append((SpanLabels.NP, np_start, noun_end))

    def _extract_noun_phrase_sequence(self):
        """
        Noun Phrase Sequence (NPS): 명사(구) 나열 구조.

        트리거: NP_CNJ rel을 가진 JC 조사(와/과/나/이나/랑/이랑) 또는 SP 쉼표.
        - NP_CNJ 노드의 head(=연결 대상 명사 어절)와
          NP_CNJ 바로 앞 명사 어절을 함께 수집
        - 같은 최종 head를 공유하는 NP_CNJ가 여럿이면 전체를 하나의 NPS로 묶음
        - 연결 대상 어절 뒤에 '등'(NNB)이 오면 포함
        - NP 스팬이 이미 추출됐으면 그 범위로 왼쪽 확장
        """
        # NP_CNJ 토큰 수집: (cnj_idx, head_idx)
        cnj_list = []
        for i in range(self.N):
            if self.rels[i] == 'NP_CNJ':
                head_idx = self.arcs[i] - 1 if self.arcs[i] > 0 else -1
                if head_idx >= 0:
                    cnj_list.append((i, head_idx))

        if not cnj_list:
            return

        # head별로 NP_CNJ 그룹핑 — 같은 head를 가리키는 CNJ들이 하나의 NPS
        from collections import defaultdict
        groups: dict = defaultdict(list)
        for cnj_idx, head_idx in cnj_list:
            groups[head_idx].append(cnj_idx)

        for head_idx, cnj_indices in groups.items():
            # NPS에 포함할 노드 범위 결정
            nodes = set()

            # 1. head 어절 수집 — head_idx의 morph chain 전체.
            #    단, head 자체가 조사(JK*/JX)이면 그 조사는 NPS 끝에서 제외
            head_eojeol_start = self._expand_eojeol_left(head_idx)
            head_is_josa = self._has_any_tag(head_idx, ('JKS','JKO','JKB','JKG','JKC','JX'))
            head_noun_end = head_idx - 1 if head_is_josa else head_idx
            for k in range(head_eojeol_start, head_noun_end + 1):
                nodes.add(k)

            # head가 '등'/'따위'(NNB) 자체인 경우: 그 앞의 명사 어절도 NPS에 포함
            # (태풍, 가뭄, 홍수 **등** 패턴에서 홍수가 rel=NP로 등을 head로 가리킴)
            _TRAILING_NNB = ('등', '따위')
            if self.tokens[head_idx] in _TRAILING_NNB and self._has_tag(head_idx, 'NNB'):
                prev = head_idx - 1
                if prev >= 0 and not self._has_any_tag(prev, ('JKS','JKO','JKB','JKG','JKC','JX')):
                    eojeol_s = prev
                    for j in range(prev - 1, -1, -1):
                        h = self.arcs[j] - 1
                        if h > j and (self.rels[j] == 'morph' or (self.rels[j] == 'NP' and h <= head_idx)):
                            eojeol_s = j
                        else:
                            break
                    for k in range(eojeol_s, prev + 1):
                        nodes.add(k)

            # head 뒤에 '등'/'따위'(NNB)가 오면 포함 — rel=morph 또는 rel=NP로 연결된 경우 모두
            k = head_noun_end + 1
            while k < self.N:
                if self.tokens[k] in _TRAILING_NNB and self._has_tag(k, 'NNB'):
                    if self.rels[k] in ('morph', 'NP') and self.arcs[k] - 1 >= head_noun_end:
                        nodes.add(k)
                    break
                if self.rels[k] == 'morph':
                    k += 1
                else:
                    break

            for cnj_idx in cnj_indices:
                # 2. JC/SP 자체 포함
                nodes.add(cnj_idx)

                # 3. JC/SP 바로 앞 명사 어절 수집
                #    — 이미 추출된 NP 스팬이 있으면 그 범위 전체 사용
                prev_end = cnj_idx - 1
                if prev_end < 0:
                    continue

                # 이미 추출된 NP 스팬 중 prev_end를 포함하는 것 찾기
                np_match = [
                    (s, e) for lbl, s, e in self.spans
                    if lbl == SpanLabels.NP and s <= prev_end <= e
                ]
                if np_match:
                    # 가장 넓은 NP 사용
                    np_match.sort(key=lambda x: x[0])
                    np_s, np_e = np_match[0]
                    for k in range(np_s, np_e + 1):
                        nodes.add(k)
                else:
                    # NP 스팬 없으면 어절 시작까지 수집 (조사 직전까지)
                    # rel=morph 또는 rel=NP이면서 head가 cnj_idx 이하를 가리키면 같은 어절 내부
                    noun_end = prev_end
                    while noun_end > 0 and self._has_any_tag(noun_end, ('JKS','JKO','JKB','JKG','JKC','JX')):
                        noun_end -= 1
                    eojeol_s = noun_end
                    for j in range(noun_end - 1, -1, -1):
                        h = self.arcs[j] - 1
                        if h > j and (self.rels[j] == 'morph' or (self.rels[j] in ('NP', 'DP') and h <= cnj_idx)):
                            eojeol_s = j
                        else:
                            break
                    for k in range(eojeol_s, noun_end + 1):
                        nodes.add(k)

            if len(nodes) < 2:
                continue

            nps_start = min(nodes)
            nps_end = max(nodes)
            self.spans.append((SpanLabels.NPS, nps_start, nps_end))

    def _extend_nps_with_leading_modifier(self):
        """A안: 나열(NPS)의 첫 접속항 앞에 관형 수식어가 붙어 있으면 그 수식어까지 NPS 범위에
        포함한다. NPS는 '명사구들의 나열'이므로, 명사구(NP)가 수식어를 포함하듯 첫 명사구의
        수식어도 나열 막대 안에 든다.
          - 관형절/관형구는 AdjP·EmC_Adj 스팬, 소유격 'X의'도 AdjP 스팬으로 이미 잡혀 있으므로
            NPS 시작 바로 앞(nps_start-1)에서 끝나는 그 스팬들을 찾아 시작점을 왼쪽으로 당긴다.
          - 관형 수식어가 여러 겹이면 반복해서 확장한다.
        NPS 추출은 _extract_adjectival보다 먼저 돌지만, 이 후처리는 AdjP가 다 나온 뒤(구문 추출
        이후)에 호출되므로 확장 근거가 되는 관형 스팬을 참조할 수 있다."""
        # 단순 관형구(AdjP)만 확장 근거로 삼는다. 내부 논항을 가진 관형절(EmC_Adj)은 나열 전체를
        # 밖에서 수식하는 경우가 많아(예: '제3세계에 대한 보건…') 접속항 수식으로 오인하면 안 된다.
        adnom = [(s, e) for lbl, s, e in self.spans if lbl == SpanLabels.ADJP]
        if not adnom:
            return
        nps_ranges = [(s, e) for lbl, s, e in self.spans if lbl == SpanLabels.NPS]
        new_spans = []
        for label, s, e in self.spans:
            if label == SpanLabels.NPS:
                start = s
                while True:
                    p = start - 1
                    # 바로 앞 형태소가 실제 관형 수식의 끝(ETM/MM/관형격 '의')이어야 한다.
                    # (그래야 '위해'(EC)·'자연재해란'(JX) 같은 비관형 어절을 잘못 끌어오지 않음)
                    if p < 0 or not (self._has_tag(p, 'ETM')
                                     or self._has_any_tag(p, ('MM', 'MMA', 'MMD', 'MMN'))
                                     or self.tokens[p] == '의'):
                        break
                    cands = [a for (a, b) in adnom if b == p and a < start]
                    if not cands:
                        break
                    a = min(cands)   # 가장 넓은 관형 수식어의 시작점
                    # 수식어가 다른 나열(NPS) 안에서 시작하면 그 나열에 딸린 외부 수식어이므로 제외
                    if any(ns <= a <= ne for (ns, ne) in nps_ranges if (ns, ne) != (s, e)):
                        break
                    # 수식어 바로 앞이 격조사(NP+격조사)면, 그 수식어는 자기 논항을 거느리고
                    # 나열 전체를 밖에서 꾸미는 관형절이다(예: '제3세계에 대한 [보건…]'). 접속항
                    # 하나에 붙은 수식어(소외된/오랜/가까운)는 앞이 격조사가 아니므로 이때만 확장.
                    if a - 1 >= 0 and self._has_any_tag(
                            a - 1, ('JKS', 'JKO', 'JKB', 'JKG', 'JKC', 'JX')):
                        break
                    start = a
                new_spans.append((label, start, e))
            else:
                new_spans.append((label, s, e))
        self.spans = new_spans

    # ========================================================================
    # 정리
    # ========================================================================

    def _deduplicate(self) -> List[Tuple[str, int, int]]:
        """중복 제거 및 정렬"""
        # 동일 span 제거
        seen = set()
        unique = []
        for label, s, e in self.spans:
            key = (label, s, e)
            if key not in seen:
                seen.add(key)
                unique.append((label, s, e))

        # 같은 라벨에서 더 큰 스팬에 완전히 포함된 작은 스팬 제거.
        # 단 EmC_Adj(관형절 내포절)는 정답 데이터가 중첩을 유지하므로
        # (예: '얼음이 아닌' ⊂ '얼음이 아닌 따뜻한 땅을 좋아하는') 포함관계 제거 대상에서 뺀다.
        _NEST_OK = {SpanLabels.EMC_ADJ}
        result = []
        for label, s, e in unique:
            if label in _NEST_OK:
                result.append((label, s, e))
                continue
            dominated = any(
                label == ol and os <= s and e <= oe and (os, oe) != (s, e)
                for ol, os, oe in unique
            )
            if not dominated:
                result.append((label, s, e))

        # 정렬: 시작 위치, 길이
        result.sort(key=lambda x: (x[1], x[2] - x[1]))

        return result


# ========================================================================
# 편의 함수
# ========================================================================

def extract_spans_klue(tokens: List[str], xpos: List[str],
                       arcs: List[int], rels: List[str]) -> List[Tuple[str, int, int]]:
    """
    KLUE-DP 관계 기반 span 추출

    Args:
        tokens: 형태소 리스트
        xpos: 품사 태그
        arcs: head indices (1-indexed)
        rels: KLUE-DP 관계 (NP_SBJ, NP_OBJ, VP, morph 등)

    Returns:
        [(label, start, end), ...] (0-indexed, end inclusive)
    """
    extractor = MorphemeSpanExtractorKLUE(tokens, xpos, arcs, rels)
    return extractor.extract()
