import re
from typing import Dict, List, Tuple

RULE_FORM_POS: Dict[Tuple[str, str], List[str] | str] = {
    # 보조 용언
    ("하", "VX"): "do",
    # 긍정 지시사(이다) (VCP)
    ("이", "VCP"): "Copula",
    # 부정 지시사(아니다) (VCN)
    ("아니", "VCN"): "Negative Copula",
    # 주격 조사 (JKS)
    ("이", "JKS"): "Subject Marker",
    ("가", "JKS"): "Subject Marker",
    ("에게", "JKS"): "Subject Marker",
    ("에서", "JKS"): "Subject Marker",
    # 보격 조사 (JKC)
    ("이", "JKC"): "Complement Marker",
    ("가", "JKC"): "Complement Marker",
    # 관형격 조사 (소유) (JKG)
    ("의", "JKG"): "Possessive Marker",
    # 목적격 조사 (JKO)
    ("을", "JKO"): "Object Marker",
    ("를", "JKO"): "Object Marker",
    ("ᆯ", "JKO"): "Object Marker",
    # --- 부사격 조사 (JKB)---
    ("에", "JKB"): "Time/Place particle ~ at/in/on/to/per / because",  # 0410_태깅_목록_의존명사.xlsx 해결안 반영 — 시간/장소 용법과 원인 용법 결합
    ("에서", "JKB"): "at/in/from",  # 위치/시점
    ("서", "JKB"): "at/in/from",    # 장소-기점/출발점
    ("에게", "JKB"): "to (a person)",
    ("한테", "JKB"): "to (a person)",
    ("께", "JKB"): "To / by (honorific)",
    ("로", "JKB"): "towards/by means of/made of/because of/as",
    ("으로", "JKB"): "towards/by means of/made of/because of/as",
    ("로부터", "JKB"): "From, starting point",
    ("으로부터", "JKB"): "From, starting point",
    ("로서", "JKB"): "As (a social position)",
    ("으로서", "JKB"): "As (a social position)",
    ("로써", "JKB"): "with/of/as for/because of",
    ("으로써", "JKB"): "with/of/as for/because of",
    ("와", "JKB"): ["COM"],
    ("과", "JKB"): ["COM"],         # 공귀(함께) 용법
    ("이랑", "JKB"): "with",
    ("처럼", "JKB"): "like",
    ("같이", "JKB"): "like",
    ("대로", "JKB"): "as/seperately",
    ("만큼", "JKB"): "as much as",
    ("에서부터", "JKB"): "Time/Place ~ from",
    ("서부터", "JKB"): "Time/Place ~ from",
    ("보고", "JKB"): "to",
    ("보다", "JKB"): "than",
    ("에다", "JKB"): "Time/Place particle ~ in/on/in addition to",
    ("에다가", "JKB"): "Time/Place particle ~ in/on/in addition to",
    ("치고", "JKB"): "as for",
    ("하고", "JKB"): "with",
    # 호격 조사 (JKV)
    ("아", "JKV"): "Hey~",
    ("야", "JKV"): "Hey~",
    ("여", "JKV"): "Hey~",
    ("이여", "JKV"): "Hey~",
    ("이시여", "JKV"): "Hey~",
    # 인용격(대표)
    ("라고", "JKQ"): "Quotation Marker",
    ("고", "JKQ"): "Quotation Marker",
    ("이라고", "JKQ"): "Quotation Marker",
    ("라며", "JKQ"): "Quotation Marker",
    ("라니", "JKQ"): "Quotation Marker",
    # --- 보조사 (Aux Particles, JX) --- 
    ("은", "JX"): "Topic Marker",
    ("는", "JX"): "Topic Marker",  # 주제
    ("도", "JX"): "also/not even",  # 추가
    ("만", "JX"): "only/just",
    ("뿐", "JX"): "only",
    ("까지", "JX"): "until/to/by/even",
    ("부터", "JX"): "since/from / After ~ing",
    ("마다", "JX"): "every",
    ("조차", "JX"): "even",
    ("마저", "JX"): "even",
    ("다가", "JX"): "Time/Place particle ~ in/on/in addition to",
    ("대로", "JX"): "as/seperately",
    ("따라", "JX"): "unusually",
    ("만큼", "JX"): "as much as",
    ("밖에", "JX"): "only",
    ("야", "JX"): "of course (emphasis)",  # 강조 용법; 호격(야JKV)과 구분
    ("이야", "JX"): "of course (emphasis)",
    ("야말로", "JX"): "Topic emphatic",
    ("이야말로", "JX"): "Topic emphatic",
    ("치고", "JX"): "as for",
    ("요", "JX"): "Polite Informal",
    ("나마", "JX"): "although it is something",
    ("이나마", "JX"): "although it is something",
    # ~(이)라야(만) — 0410_태깅_목록_의존명사.xlsx 해결안 반영: 단일 JX 토큰 형태
    ("이라야", "JX"): "only if / not unless",
    ("이라야만", "JX"): "only if / not unless",
    ("라야", "JX"): "only if / not unless",
    ("라야만", "JX"): "only if / not unless",
    ("란", "JX"): "Topic Marker",  # 이란/란 — 강조 포함 topic marker
    ("이란", "JX"): "Topic Marker",
    ("커녕", "JX"): "far from it",
    ("ㄴ커녕", "JX"): "far from it",
    ("는커녕", "JX"): "far from it",
    ("은커녕", "JX"): "far from it",
    # --- JX로도 나오는 'or' 계열 (분석기마다 JX/JC 혼용 대비) ---
    ("나", "JX"): ["or"],
    ("이나", "JX"): "As many as / rather / approximately",  # 양보/정도 용법 (or 용법은 JC)
    ("든가", "JX"): "or",
    ("든지", "JX"): "or",
    ("이라든가", "JX"): "or",
    ("이라든지", "JX"): "or",
    ("라든가", "JX"): "or",
    ("라든지", "JX"): "or",
    # --- 접속조사 (JC) ---
    ("와", "JC"): "and",
    ("과", "JC"): "and",
    ("랑", "JC"): "and",
    ("이랑", "JC"): "and",
    ("하고", "JC"): "and",
    ("및", "JC"): "and",
    ("나", "JC"): "or",
    ("이나", "JC"): "or",
    ("거나", "JC"): "or",
    ("든지", "JC"): "or",
    ("든가", "JC"): "or",
    ("이라든가", "JC"): "or",
    ("이라든지", "JC"): "or",
    ("라든가", "JC"): "or",
    ("라든지", "JC"): "or",
    ("치고", "JC"): "as for",
    # --- 선어말어미 (EP) ---
    ("았", "EP"): "Past Tense, ~ed",
    ("었", "EP"): "Past Tense, ~ed",
    ("였", "EP"): "Past Tense, ~ed",
    ("었었", "EP"): "Past Tense",   # 과거완료
    ("았었", "EP"): "Past Tense",
    ("겠", "EP"): "Probably would/will/Polite statement",
    ("으시", "EP"): "Honoring the subject of a sentence",
    ("시", "EP"): "Honoring the subject of a sentence",
    ("더", "EP"): ["RET"],
    # --- 종결어미 (EF) : 대표 표면형들 ---
    ("다", "EF"): "Plain-style Statement",
    ("ㄴ다", "EF"): "Plain-style Statement",
    ("는다", "EF"): "Plain-style Statement",
    ("요", "EF"): "Polite Informal",
    ("까", "EF"): ["INT"],
    ("니", "EF"): "Plain-style Question",
    ("으니", "EF"): "Plain-style Question",
    ("냐", "EF"): "Plain-style Question",
    ("으냐", "EF"): "Plain-style Question",
    ("느냐", "EF"): "Plain-style Question",
    ("네", "EF"): "Simple Statement / Exclamation",
    ("네요", "EF"): "Polite-style Exclamation",
    ("군", "EF"): "Exclamation",
    ("군요", "EF"): "Exclamation",
    ("는군", "EF"): "Exclamation",
    ("는군요", "EF"): "Exclamation",
    ("라", "EF"): ["IMP"],
    ("자", "EF"): "Let's",
    ("마라", "EF"): ["PROH"],
    ("세요", "EF"): "Polite Command",
    ("으세요", "EF"): "Polite Command",
    ("습니다", "EF"): "Formal Statement",
    ("ᆸ니다", "EF"): "Formal Statement",
    ("습니까", "EF"): "Formal Question",
    ("ᆸ니까", "EF"): "Formal Question",
    ("지요", "EF"): "Asking for agreement or confirmation / making a suggestion / soft question",
    ("죠", "EF"): "Asking for agreement or confirmation / making a suggestion / soft question",
    ("래", "EF"): "Quoted Statement",
    ("으래", "EF"): "Quoted Statement",
    ("라니", "EF"): "asking back / showing surprise",
    ("리라", "EF"): "shall/will",
    ("으리라", "EF"): "shall/will",
    ("려고", "EF"): "Intimate style sentence-ending",
    ("ᆸ시오", "EF"): "Formal Command",
    ("거든", "EF"): "because / actually",
    ("거라", "EF"): "Plain-style Command",
    ("게", "EF"): "Familiar-style command / trying to / guess / ironic expression",
    ("구나", "EF"): "Exclamation",
    ("로구나", "EF"): "Exclamation",
    ("구먼", "EF"): "Exclamation / rebuke",
    ("는구먼", "EF"): "Exclamation / rebuke",
    ("ᆫ가", "EF"): "Familiar-style Question/Exclamation",
    ("ㄴ가", "EF"): "Familiar-style Question/Exclamation",
    ("는가", "EF"): "Familiar-style Question/Exclamation",
    ("은가", "EF"): "Familiar-style Question/Exclamation",
    ("ᆫ다고", "EF"): "Predicate-ending",
    ("ㄴ다고", "EF"): "Predicate-ending",
    ("는다고", "EF"): "Predicate-ending",
    ("ᆫ다더라", "EF"): "hearsay report",
    ("ㄴ다더라", "EF"): "hearsay report",
    ("는다더라", "EF"): "hearsay report",
    ("ᆫ다던가", "EF"): "Hearsay Question",
    ("ㄴ다던가", "EF"): "Hearsay Question",
    ("는다던가", "EF"): "Hearsay Question",
    ("ᆫ다던데", "EF"): "Hearsay Remark",
    ("ㄴ다던데", "EF"): "Hearsay Remark",
    ("는다던데", "EF"): "Hearsay Remark",
    ("ᆫ다면서", "EF"): "Hearsay Question",
    ("ㄴ다면서", "EF"): "Hearsay Question",
    ("는다면서", "EF"): "Hearsay Question",
    ("ᆫ다지", "EF"): "Quoted Question",
    ("ㄴ다지", "EF"): "Quoted Question",
    ("는다지", "EF"): "Quoted Question",
    ("ᆫ답니다", "EF"): "Quoted Statement (short form)",
    ("ㄴ답니다", "EF"): "Quoted Statement (short form)",
    ("는답니다", "EF"): "Quoted Statement (short form)",
    ("답니다", "EF"): "Quoted Statement (short form)",
    ("나", "EF"): "Soft Question",
    ("나요", "EF"): "Soft Question",
    ("냬", "EF"): "Quoted Question",
    ("너라", "EF"): "Plain-style Command",
    ("는다니", "EF"): "asking back / showing surprise",
    ("는대", "EF"): "Quoted Statement (short form)",
    ("는데", "EF"): "Exclamation / Question",
    ("다니", "EF"): "asking back / showing surprise",
    ("다오", "EF"): "please / give me / Soft Statement",
    ("단다", "EF"): "soft statement",
    ("대", "EF"): "Quoted Statement",
    ("더군", "EF"): "learned from an experience",
    ("더군요", "EF"): "learned from an experience",
    ("더냐", "EF"): "asking about listener's personal experience",
    ("더니", "EF"): "I saw... (and now)",
    ("더라", "EF"): "found that",
    ("더라고", "EF"): "I recall that",
    ("던가", "EF"): "asking/assuming about the past fact",
    ("던데", "EF"): "recollection",
    ("ᆯ게", "EF"): "promise ~ like futures",
    ("ㄹ게", "EF"): "promise ~ like futures",
    ("을게", "EF"): "promise ~ like futures",
    ("ᆯ걸", "EF"): "I guess / I should have",
    ("ㄹ걸", "EF"): "I guess / I should have",
    ("을걸", "EF"): "I guess / I should have",
    ("ᆯ까", "EF"): "I wonder if / Shall we~?",
    ("ㄹ까", "EF"): "I wonder if / Shall we~?",
    ("을까", "EF"): "I wonder if / Shall we~?",
    ("ᆯ라", "EF"): "I'm afraid~",
    ("ㄹ라", "EF"): "I'm afraid~",
    ("을라", "EF"): "I'm afraid~",
    ("ᆯ래", "EF"): "will(want to)",
    ("ㄹ래", "EF"): "will(want to)",
    ("을래", "EF"): "will(want to)",
    ("라고", "EF"): "Emphatic statement / asking back",
    ("라니까", "EF"): "Emphatic statement",
    ("란다", "EF"): "(formal, highly addressee-lowering) sentence ending",
    ("이란다", "EF"): "(formal, highly addressee-lowering) sentence ending",
    ("려나", "EF"): "I guess~",
    ("렵니까", "EF"): "Would you~?",
    ("렵니다", "EF"): "I would like to~",
    ("마", "EF"): "I'll~",
    ("세", "EF"): "Familiar-style proposal",
    ("아", "EF"): "Casual Ending",
    ("어", "EF"): "Casual Ending",
    ("아요", "EF"): "Polite Casual Ending",
    ("어요", "EF"): "Polite Casual Ending",
    ("여요", "EF"): "Polite Casual Ending",
    ("아라", "EF"): "Plain-style Command",
    ("어라", "EF"): "Plain-style Command",
    ("으라", "EF"): "Plain-style Command",
    ("어야지", "EF"): "Determination; reasonableness",
    ("아야지", "EF"): "Determination; reasonableness",
    ("오", "EF"): "Semiformal Statement",
    ("소", "EF"): "Semiformal Statement",
    ("으라면서", "EF"): "You said that~",
    ("라면서", "EF"): "You said that~",
    ("는걸", "EF"): "I guess / Actually",
    ("ᆫ걸", "EF"): "I guess / Actually",
    ("ㄴ걸", "EF"): "I guess / Actually",
    ("ᆸ시다", "EF"): "Let's, shall we",
    ("ᆸ시오", "EF"): "Formal Command",
    ("읍시다", "EF"): "Let's, shall we",
    ("자니까", "EF"): "Let's",
    ("잖아", "EF"): "checking or correcting",
    ("잖아요", "EF"): "checking or correcting",
    ("지", "EF"): "Asking for agreement or confirmation / making a suggestion / soft question",
    ("지요", "EF"): "Asking for agreement or confirmation / making a suggestion / soft question",
    ("다니까", "EF"): "Question Ending",
    # --- 연결어미 (EC) : 흔한 표면형 ---
    ("고", "EC"): "and",
    ("라고", "EC"): "that",  # 인용절 트리거 (간접 인용) — 정답 데이터 기준
    ("냐고", "EC"): "asking if",
    ("자고", "EC"): "suggesting that",
    ("서", "EC"): ["and_then"],
    ("지만", "EC"): "but",
    ("지", "EC"): "but",             # 부정/대조 연결 용법
    ("지마는", "EC"): "but",
    ("는데", "EC"): "even though",   # 엑셀 업데이트: but/and → even though
    ("ㄴ데", "EC"): "even though",
    ("은데", "EC"): "even though",
    ("면", "EC"): "if, when",        # 엑셀 업데이트: if → if, when
    ("으면", "EC"): "if, when",
    ("으니", "EC"): "because/when",
    ("니", "EC"): "because/when",
    ("으며", "EC"): "and/while doing",
    ("며", "EC"): "and/while doing",
    ("면서", "EC"): "while",
    ("으면서", "EC"): "while",
    ("도록", "EC"): "so that / to the extent that",
    ("게", "EC"): "~ly",             # 엑셀 업데이트: so that → ~ly (부사형)
    ("자마자", "EC"): "As soon as",
    ("거나", "EC"): "or / no matter whether or",
    ("든지", "EC"): "either or / whether or not / no matter",
    ("든", "EC"): "either or / whether or not / no matter",
    ("더니", "EC"): "observed",
    ("다가", "EC"): "while doing / Caution / Switch of actions",  # 엑셀 업데이트
    ("다", "EC"): "while doing / Caution / Switch of actions",
    ("느라", "EC"): ["because_doing"],
    ("느라고", "EC"): "because / in order to",  # 엑셀 업데이트: because → because / in order to
    ("라서", "EC"): "and/because",   # 엑셀 업데이트: because → and/because
    ("아서", "EC"): "because/and/to",
    ("어서", "EC"): "because/and/to",
    ("ᆯ망정", "EC"): "even if",
    ("ㄹ망정", "EC"): "even if",
    ("을망정", "EC"): "even if",
    ("ᆯ지", "EC"): "whether or not",
    ("ㄹ지", "EC"): "whether or not",
    ("을지", "EC"): "whether or not",
    ("ᆯ지언정", "EC"): "even if",
    ("ㄹ지언정", "EC"): "even if",
    ("을지언정", "EC"): "even if",
    ("ᆯ지라도", "EC"): "even though",
    ("ㄹ지라도", "EC"): "even though",
    ("을지라도", "EC"): "even though",
    ("ᆯ수록", "EC"): "the more",    # 엑셀 추가
    ("ㄹ수록", "EC"): "the more",
    ("을수록", "EC"): "the more",
    ("ᆯ라치면", "EC"): "whenever one is about to",
    ("ㄹ라치면", "EC"): "whenever one is about to",
    ("을라치면", "EC"): "whenever one is about to",
    ("으러", "EC"): "in order to",
    ("러", "EC"): "in order to",
    ("으려고", "EC"): "intention / doubt",
    ("려고", "EC"): "intention / doubt",
    ("으려다가", "EC"): "intended to ~ but",
    ("려다가", "EC"): "intended to ~ but",
    ("려도", "EC"): "even if I try to (verb)",  # 엑셀 업데이트
    ("으려도", "EC"): "even if I try to (verb)",
    ("으려면", "EC"): "if you intend to / until something happens",  # 엑셀 업데이트
    ("려면", "EC"): "if you intend to / until something happens",
    ("으리라고", "EC"): "will / guess",  # 엑셀 업데이트
    ("리라고", "EC"): "will / guess",
    ("므로", "EC"): "because",
    ("으므로", "EC"): "because",
    ("거니와", "EC"): "As well as / as",
    ("거든", "EC"): "if / even",
    ("건대", "EC"): "as I see/heard/wish/promise...",
    ("건만", "EC"): "even though",
    ("고도", "EC"): "even after ~ing / and (additionally)",
    ("고서", "EC"): "after doing / if not",
    ("고자", "EC"): "In order to",
    ("기로서니", "EC"): "even if",
    ("기에", "EC"): "because",
    ("길래", "EC"): "because",
    ("는다거나", "EC"): "for example ~ or ~",
    ("ᆫ다거나", "EC"): "for example ~ or ~",
    ("ㄴ다거나", "EC"): "for example ~ or ~",
    ("는다고", "EC"): "as / as it is said",
    ("ᆫ다고", "EC"): "as / as it is said",
    ("ㄴ다고", "EC"): "as / as it is said",
    ("ᆫ다기에", "EC"): "I heard that ... so ...",
    ("ㄴ다기에", "EC"): "I heard that ... so ...",
    ("는다기에", "EC"): "I heard that ... so ...",
    ("느니", "EC"): "(rather) than",
    ("ᆫ다면", "EC"): "if (low possibility)",
    ("ㄴ다면", "EC"): "if (low possibility)",
    ("는다면", "EC"): "if (low possibility)",
    ("다면", "EC"): "if (low possibility)",
    ("ᆫ다면서", "EC"): "saying that",
    ("ㄴ다면서", "EC"): "saying that",
    ("는다면서", "EC"): "saying that",
    ("ᆫ바", "EC"): "because",
    ("ㄴ바", "EC"): "because",
    ("는바", "EC"): "because",
    ("ᆫ지", "EC"): "vague doubt/judgement",
    ("ㄴ지", "EC"): "vague doubt/judgement",
    ("는지", "EC"): "vague doubt/judgement",
    ("ᆫ들", "EC"): "even though someone or something",
    ("ㄴ들", "EC"): "even though someone or something",
    ("은들", "EC"): "even though someone or something",
    ("ᆫ다마는", "EC"): "but",
    ("ㄴ다마는", "EC"): "but",
    ("는다마는", "EC"): "but",
    ("다만", "EC"): "but",
    ("ᆫ다만", "EC"): "but",
    ("ㄴ다만", "EC"): "but",
    ("는다만", "EC"): "but",
    ("다시피", "EC"): "as / almost",
    ("더라도", "EC"): "even if",
    ("던지", "EC"): "so… that…",
    ("되", "EC"): "even though / someone says",
    ("든가", "EC"): "or / no matter",
    ("듯이", "EC"): "as if",
    ("듯", "EC"): "as if",
    ("라", "EC"): "because/but",
    ("라더니", "EC"): "I heard that ... so ...",  # 오탈자 수정
    ("라면", "EC"): "if",
    ("으라면", "EC"): "if",
    ("랴", "EC"): "And (doing several things at the same time)",
    ("로되", "EC"): "even though",
    ("려야", "EC"): "want to ~ but",
    ("ᆯ래야", "EC"): "want to ~ but",
    ("ㄹ래야", "EC"): "want to ~ but",
    ("을래야", "EC"): "want to ~ but",
    ("자니까", "EC"): "as one suggests",
    ("자니", "EC"): "on one's suggestion",
    ("자면", "EC"): "If I mean to (verb) / If someone asks me to (verb) / for something to (verb)",
    ("다고", "EC"): "as / as it is said",
    ("어", "EC"): "and/because",
    ("어야지", "EC"): "only if",
    ("어도", "EC"): "even though",
    ("어다", "EC"): "and",
    ("어다가", "EC"): "and",
    ("나", "EC"): "however/or",
    ("으나", "EC"): "however/or",
    ("으니까", "EC"): "because / doing something and realized",
    ("니까", "EC"): "because / doing something and realized",
    ("자", "EC"): "As / No matter / And",
    # --- 명사 파생 접미사 (XSN) ---
    ("들", "XSN"): "pluralizer",
    # --- 명사형 전성 어미 (ETN) ---
    ("기", "ETN"): "Noun-forming",
    ("ㅁ", "ETN"): "Noun-forming",
    ("음", "ETN"): "Noun-forming",
    # --- 관형형 전성 어미 (ETM) ---
    ("ㄴ", "ETM"): "Adjective-forming",
    ("은", "ETM"): "Adjective-forming",
    ("는", "ETM"): "Adjective-forming",
    ("ㄹ", "ETM"): "Adjective-forming",
    ("을", "ETM"): "Adjective-forming",
    ("던", "ETM"): "Adjective-forming",
    ("으리라는", "ETM"): "that one will (verb) / that one would (verb/adjective)",
    ("리라는", "ETM"): "that one will (verb) / that one would (verb/adjective)",
    ("ᆯ", "ETM"): "Adjective-forming",
    ("ᆫ", "ETM"): "Adjective-forming",  # 관형사형
    # --- 보정: 구두점 등 ---
    (".", "SF"): ["."],
    (",", "SP"): [","],
    ("?", "SF"): ["?"],
    ("!", "SF"): ["!"],
    # --- 의존 명사 (NNB) — 0410_태깅_목록.xlsx '의존 명사' 시트 의미 정보 기준 ---
    # bound noun
    ("척", "NNB"): "pretending to do (or be) something",
    ("탓", "NNB"): "the cause of a negative outcome; fault/blame",
    # 탓/NNB는 KLUE-DP 학습 코퍼스(supar_morph_dp/klue-*-morph.conllu) 기준 실사용례가 없음 —
    # Kiwi는 '탓'을 항상 NNG로 태깅하므로(코퍼스 19/19건 NNG) 위 NNB 항목은 실제로 발동하지 않는다.
    # 같은 의미를 NNG 키로도 등록해 실제 입력 태그에서 매칭되도록 함
    ("탓", "NNG"): "the cause of a negative outcome; fault/blame",
    ("덕", "NNB"): "cause of a positive outcome; thanks to",
    # 덕도 코퍼스에서 NNG 11/11건 — 탓과 동일한 사유로 NNG 키 병행 등록
    ("덕", "NNG"): "cause of a positive outcome; thanks to",
    ("체", "NNB"): "pretending to do or be something; acting as if something is true",
    ("것", "NNB"): "a thing, fact, act, or state; used to turn a clause into a noun",
    ("거", "NNB"): "a thing, fact, act, or state; used to turn a clause into a noun",  # 구어체 '것'
    ("걸", "NNB"): "a thing, fact, act, or state; used to turn a clause into a noun",  # 구어체 '것을'
    ("바", "NNB"): "the matter or fact stated before it",
    ("줄", "NNB"): "the way, fact, or expectation expressed by what comes before it",
    ("데", "NNB"): "a place, situation, or purpose related to what comes before it",
    ("지", "NNB"): "the time since an action or state began",
    ("수", "NNB"): "a way, ability, or possibility to do something",
    ("따름", "NNB"): "only that; nothing more than the preceding statement",
    ("뿐", "NNB"): "only the preceding action or state; nothing else",
    ("만큼", "NNB"): "the same degree, amount, or extent as what comes before it",
    ("대로", "NNB"): "the same way as what comes before it",
    ("만치", "NNB"): "the same degree, amount, or extent as what comes before it",
    ("적", "NNB"): "a time or experience of doing something",
    ("밖", "NNB"): "no choice but to do or be what comes before it",
    ("때문", "NNB"): "the reason or cause of a result",
    ("터", "NNB"): "an expected situation, intention, or condition",
    ("따위", "NNB"): "things of that kind; and the like, often with a dismissive tone",
    ("동안", "NNB"): "a period of time while something happens or continues",
    # 동안도 코퍼스에서 NNG 155/155건 — 탓/덕과 동일 사유로 NNG 키 병행 등록
    ("동안", "NNG"): "a period of time while something happens or continues",
    ("전", "NNB"): "the time before, or the front of, something",
    # 전은 코퍼스에서 NNB가 5/432건(1%)뿐이고 NNG/MMD가 압도적 — NNG 키 병행 등록
    ("전", "NNG"): "the time before, or the front of, something",
    ("후", "NNB"): "the time after something, or the later/back part of something",
    # 후도 코퍼스에서 NNG 134/134건 — 탓/덕과 동일 사유로 NNG 키 병행 등록
    ("후", "NNG"): "the time after something, or the later/back part of something",
    ("중", "NNB"): "the time when something is happening or in progress",
    # counter noun
    ("개", "NNB"): "general counting unit",
    ("명", "NNB"): "counting unit for people",
    ("분", "NNB"): "counting unit for people (honorific)",
    ("마리", "NNB"): "counting unit for animals",
    ("권", "NNB"): "counting unit for books",
    ("장", "NNB"): "counting unit for paper / flat objects",
    ("통", "NNB"): "counting unit for containers/cans",
    ("켤레", "NNB"): "counting unit for pairs (shoes/socks)",
    ("조각", "NNB"): "counting unit for pieces/slices",
    ("잔", "NNB"): "counting unit for cups/glasses (drinks)",
    ("병", "NNB"): "counting unit for bottles",
    ("리터", "NNB"): "liter",
    ("킬로그램", "NNB"): "kilogram",
    ("원", "NNB"): "currency unit (Korean won)",
    ("회", "NNB"): "counting unit for times/events (formal)",
    ("번", "NNB"): "counting unit for turns/occasions",
    ("살", "NNB"): "counting unit for age (years old)",
    ("세", "NNB"): "counting unit for years (age) (very formal/older form)",
    ("쌍", "NNB"): "counting unit for pairs",
    ("벌", "NNB"): "counting unit for clothing",
    ("쪽", "NNB"): "page",
    ("대", "NNB"): "counting unit for vehicles/machines",
    ("칸", "NNB"): "counting unit for sections/grids",
}

# 정규식 풀백
RULE_REGEX_POS = {
    "EF": [
        (re.compile(r".*습니까$"), "Formal Question"),
        (re.compile(r".*ᆸ니까$"), "Formal Question"),
        (re.compile(r".*습니다$"), "Formal Statement"),
        (re.compile(r".*ᆸ니다$"), "Formal Statement"),
        (re.compile(r".*세요$"), "Polite Command"),
        (re.compile(r".*군요$"), "Exclamation"),
        (re.compile(r".*(지요|죠)$"), "Asking for agreement or confirmation / making a suggestion / soft question"),
        (re.compile(r".*까$"), "INT"),
        (re.compile(r".*마라$"), "PROH"),
        (re.compile(r"^예$"), "Polite Informal"),  # 이다 어간 활용 (-거예요/-것이에요의 '예')
        (re.compile(r".*다$"), "Plain-style Statement"),
        (re.compile(r".*요$"), "Polite Informal"),
        (re.compile(r".*으(니|냐)$"), "Plain-style Question"),
        (re.compile(r".*(네|군)$"), "Exclamation"),
        (re.compile(r".*래$"), "Quoted Statement"),
    ],
    "EC": [
        (re.compile(r".*는데도?$"), "even though"),
        (re.compile(r".*는데$"), "even though"),
        (re.compile(r".*(으)?니까$"), "because / doing something and realized"),
        (re.compile(r".*서$"), "and then"),
        (re.compile(r".*며$"), "and/while doing"),
        (re.compile(r".*도록$"), "so that / to the extent that"),
        (re.compile(r".*(거나|든지)$"), "or"),
        (re.compile(r".*자마자$"), "as soon as"),
        (re.compile(r".*느라$"), "because_doing"),
        (re.compile(r".*더니$"), "but_then"),
        (re.compile(r"(려야|ㄹ래야|을래야)$"), "want to ~ but"),
        (re.compile(r".*(ᆫ다면서|는다면서|ᆫ지다면서)$"), "saying that"),
        (re.compile(r".*(ᆫ바|는바|ㄴ바)$"), "because"),
        (re.compile(r".*(ᆫ지|는지|ㄴ지)$"), "vague doubt/judgement"),
        (re.compile(r".*(아|어|여)$"), "and/because"),
        (re.compile(r".*(아|어|여)야지$"), "only if"),
        (re.compile(r".*(아|어|여)야$"), "only if"),
        (re.compile(r".*(아|어|여)도$"), "even though"),
        (re.compile(r".*(아|어|여)다(가)?$"), "and"),
        (re.compile(r".*(으)?나$"), "however/or"),
        (re.compile(r"^자$"), "as / no matter / and"),
        (re.compile(r".*(ㄴ|은)들$"), "even though"),
        (re.compile(r".*(ㄹ|을)라치면$"), "whenever one is about to"),
        (re.compile(r".*(ㄹ|을)수록$"), "the more"),
        (re.compile(r".*(ㄹ|을)망정$"), "even if"),
        (re.compile(r".*(ㄹ|을)지라도$"), "even though"),
        (re.compile(r".*(ㄹ|을)지언정$"), "even if"),
    ],
    "EP": [
        (re.compile(r".*(았|었|였)$"), "Past Tense"),
        (re.compile(r".*겠$"), "Would/Will"),
        (re.compile(r".*시$"), "Honorific"),
        (re.compile(r".*더$"), "RET"),
    ],
    "ETM": [
        (re.compile(r".*(ㄴ|은)$"), "ADN.PST"),
        (re.compile(r".*는$"), "ADN.PRS"),
        (re.compile(r".*(ㄹ|을)$"), "ADN.FUT"),
        (re.compile(r".*던$"), "ADN.RET"),
    ],
    "ETN": [(re.compile(r".*(기|ㅁ|음)$"), "NMLZ")],
    "JC": [
        (re.compile(r".*(나|이나|거나|든지|라든가|라든지)$"), "or"),
        (re.compile(r".*(와|과|랑|및)$"), "and"),
    ],
    "JX": [
        (re.compile(r".*(나|이나|라든가|라든지)$"), "or"),
        (re.compile(r".*(은|는)$"), "Topic Marker"),
        (re.compile(r".*(란|이란)$"), "Topic Marker"),
        (re.compile(r".*(은|는|ㄴ)?커녕$"), "far from it"),
    ],
    "JKB": [
        (re.compile(r".*께$"), "To / by (honorific)"),
        (re.compile(r".*(으로|로)$"), "towards/by means of/made of/because of/as"),
        (re.compile(r".*(처럼|같이)$"), "like"),
        (re.compile(r".*대로$"), "as/seperately"),
        (re.compile(r".*(으로부터|로부터)$"), "From, starting point"),
        (re.compile(r".*(으로서|로서)$"), "As (a social position)"),
        (re.compile(r".*(으로써|로써)$"), "with/of/as for/because of"),
        (re.compile(r".*(이나마|나마)$"), "although it is something"),
        (re.compile(r".*다가$"), "time/place particle ~ in/on/in addition to"),
        (re.compile(r".*(은|는|ㄴ)?커녕$"), "far from it"),
        (re.compile(r".*(와|과)$"), "COM"),
    ],
}
ANY = "*"
# 복합 패턴 우선 매칭
COMPOUND_JKB_TO = {
    ("에게", "JKB", "로", "JKB"): ("to/towards", "Particle"),
    ("에게", "JKB", "으로", "JKB"): ("to", "Particle"),
    ("한테", "JKB", "로", "JKB"): ("to", "Particle"),
    ("한테", "JKB", "으로", "JKB"): ("to", "Particle"),
    ("께", "JKB", "로", "JKB"): ("to", "Particle"),
    ("께", "JKB", "으로", "JKB"): ("to", "Particle"),
    ("이", "VCP", "라든가", "EC"): ("or", "Particle"),
    ("이", "VCP", "라든지", "EC"): ("or", "Particle"),
    ("이", "VCP", "라고", "EC"): ("as/even", "Particle"),
    ("이", "VCP", "라고", "EC", "ᆫ", "JX"): ("as/even", "Particle"),
    ("이", "VCP", "라고", "EC", "ㄴ", "JX"): ("as/even", "Particle"),
    ("이", "VCP", "라고", "EC", "는", "JX"): ("as/even", "Particle"),
    ("이", "VCP", "라도", "EC"): ("even/if", "Particle"),
    ("이", "VCP", "라야", "EF"): ("only if", "Particle"),
    ("이", "VCP", "라야", "EF", "만", "JX"): ("only if", "Particle"),
    ("이", "VCP", "라야", "EC"): ("only if", "Particle"),
    ("으려", "EC", "야", "JX"): ("want to ~ but", "Connector"),
    ("려", "EC", "야", "JX"): ("want to ~ but", "Connector"),
    ("을래", "EC", "야", "JX"): ("want to ~ but", "Connector"),
    ("ㄹ래", "EC", "야", "JX"): ("want to ~ but", "Connector"),
    ("으려", "EC", "다가", "EC"): ("intended to ~ but", "Connector"),
    ("으려", "EC", "도", "JX"): ("even if I try to", "Connector"),
    ("려", "EC", "도", "JX"): ("even if I try to", "Connector"),
    ("으리라", "EF", "고", "EC"): ("will / guess", "Sentence end"),
    ("고", "EC", "도", "JX"): ("even after ~ing / and (additionally)", "Connector"),
    ("기", "ETN", "에", "JKB"): ("because", "Connector"),
    ("뿐", "NNB", "더러", ANY): ("and furthermore", "Particle"),
    ("ᆯ", "ETM", "뿐", "NNB", "더러", "JKB"): ("and furthermore", "Particle"),
    ("ㄹ", "ETM", "뿐", "NNB", "더러", "JKB"): ("and furthermore", "Particle"),
    ("을", "ETM", "뿐", "NNB", "더러", "JKB"): ("and furthermore", "Particle"),
    # (으)ㄹ수록
    ("ᆯ수록", ANY): ("the more", "Connector"),
    ("을수록", "EC"): ("the more", "Connector"),
    # ㄴ다거나/는다거나
    ("ᆫ다", ANY, "거나", "EC"): ("for example ~ or ~", "Connector"),
    ("는다", ANY, "거나", "EC"): ("for example ~ or ~", "Connector"),
    ("ᆫ다", "EC", "거나", "JX"): ("for example ~ or ~", "Connector"),
    ("ㄴ다", "EC", "거나", "JX"): ("for example ~ or ~", "Connector"),
    ("는다", "EC", "거나", "JX"): ("for example ~ or ~", "Connector"),
    # (으)ㄹ지라도
    ("ᆯ지라도", ANY): ("even though", "Connector"),
    ("을지라도", ANY): ("even though", "Connector"),
    ("ᆫ다고", "EC"): ("as / as it is said", "Connector"),
    # ㄴ다기에/는다기에 복합 분석형
    ("ᆫ다", "EC", "기에", "EC"): ("I heard that ... so ...", "Connector"),
    ("ㄴ다", "EC", "기에", "EC"): ("I heard that ... so ...", "Connector"),
    ("는다", "EC", "기에", "EC"): ("I heard that ... so ...", "Connector"),
    ("ᆫ다기", "ETN", "에", "JKB"): ("I heard that ... so ...", "Connector"),
    ("ㄴ다기", "ETN", "에", "JKB"): ("I heard that ... so ...", "Connector"),
    ("는다기", ANY, "에", "JKB"): ("I heard that ... so ...", "Connector"),
    # --- advz: 형용사/형용파생 + 게 → Adverb ---
    (ANY, "VA", "게", "EC"): ("ADVZ", "Adverb"),
    (ANY, "XSA", "게", "EC"): ("ADVZ", "Adverb"),
    # --- so_that: 게 + 사역 트리거 (하/만들/시키) ---
    # 주의: '게 되다'는 보조용언(Auxiliary Verb)이므로 여기서 제외 — ② 패턴이 처리
    ("게", "EC", "하", "VV"): ("so_that", "Connector"),
    ("게", "EC", "만들", "VV"): ("so_that", "Connector"),
    ("게", "EC", "시키", "VV"): ("so_that", "Connector"),
    ("ᆫ다", ANY, "기에", ANY): ("I heard that ... so ...", "Connector"),
    ("는다", ANY, "기에", "EC"): ("I heard that ... so ...", "Connector"),
    ("ᆫ다마는", ANY): ("but", "Connector"),
    ("ᆫ", "ETM", "바", "NNB"): ("because", "Connector"),
    ("는", "ETM", "바", "NNB"): ("because", "Connector"),
    ("ᆫ", "ETM", "지", "NNB"): ("vague doubt/judgement", "Connector"),
    ("는", "ETM", "지", "NNB"): ("vague doubt/judgement", "Connector"),
    ("다가", "EC", "는", "JX"): ("caution/switch of actions", "Connector"),
    ("라", ANY, "야", "JX"): ("only if", "Connector"),
    # ③ Tense Form Pattern — ②보다 앞에 배치 (PDF §2 주의: '-고 있'은 Tense 우선)
    (ANY, "ETM", "것", "NNB", "이", "VCP"): ("Future tense", "Tense Form"),     # -ㄹ/을 + 것 + 이
    (ANY, "ETM", "거", "NNB", "이", "VCP"): ("Future tense", "Tense Form"),     # 구어 -ㄹ/을 + 거 + 이
    ("고", "EC", "있", "VX"): ("Progressive tense", "Tense Form"),              # -고 + 있
    ("고", "EC", "계시", "VX"): ("Progressive tense", "Tense Form"),            # -고 + 계시 (존경법 진행형, '있'의 높임)
    ("ᆫ", "ETM", "중", "NNB", "이", "VCP"): ("Progressive tense", "Tense Form"), # -ㄴ/는 + 중 + 이 (PDF §3: 과거 은/ㄹ 등은 제외)
    ("ㄴ", "ETM", "중", "NNB", "이", "VCP"): ("Progressive tense", "Tense Form"),
    ("는", "ETM", "중", "NNB", "이", "VCP"): ("Progressive tense", "Tense Form"),
    # ② 보조용언 결합 패턴 (PDF 형태소 표시 규칙 §2) — 값: (gloss, "Auxiliary Verb")
    ("어", "EC", "주", "VX"): ("do something for someone", "Auxiliary Verb"),
    ("아", "EC", "주", "VX"): ("do something for someone", "Auxiliary Verb"),
    ("여", "EC", "주", "VX"): ("do something for someone", "Auxiliary Verb"),
    ("어", "EC", "드리", "VX"): ("do something for someone (honorific)", "Auxiliary Verb"),
    ("아", "EC", "드리", "VX"): ("do something for someone (honorific)", "Auxiliary Verb"),
    ("여", "EC", "드리", "VX"): ("do something for someone (honorific)", "Auxiliary Verb"),
    ("어", "EC", "버리", "VX"): (None, "Auxiliary Verb"),
    ("아", "EC", "버리", "VX"): (None, "Auxiliary Verb"),
    ("여", "EC", "버리", "VX"): (None, "Auxiliary Verb"),
    ("어", "EC", "놓", "VX"): ("leave as is", "Auxiliary Verb"),
    ("아", "EC", "놓", "VX"): ("leave as is", "Auxiliary Verb"),
    ("여", "EC", "놓", "VX"): ("leave as is", "Auxiliary Verb"),
    ("고", "EC", "싶", "VX"): ("want to", "Auxiliary Verb"),
    ("어야", "EC", "하", "VX"): ("have to", "Auxiliary Verb"),
    ("아야", "EC", "하", "VX"): ("have to", "Auxiliary Verb"),
    ("여야", "EC", "하", "VX"): ("have to", "Auxiliary Verb"),
    ("어", "EC", "나가", "VX"): ("Gradually Do/Become", "Auxiliary Verb"),
    ("아", "EC", "나가", "VX"): ("Gradually Do/Become", "Auxiliary Verb"),
    ("여", "EC", "나가", "VX"): ("Gradually Do/Become", "Auxiliary Verb"),
    ("려고", "EC", "하", "VX"): ("Intend to/be about to", "Auxiliary Verb"),
    ("으려고", "EC", "하", "VX"): ("Intend to/be about to", "Auxiliary Verb"),
    # -어/아/여 + 달(VX): Kiwi가 '달라'를 '달+라'로 분리함
    ("어", "EC", "달", "VX"): ("ask someone to", "Auxiliary Verb"),
    ("아", "EC", "달", "VX"): ("ask someone to", "Auxiliary Verb"),
    ("여", "EC", "달", "VX"): ("ask someone to", "Auxiliary Verb"),
    ("곤", "EC", "하", "VX"): ("used to", "Auxiliary Verb"),
    ("게", "EC", "되", "VV"): (None, "Auxiliary Verb"),                           # -게 되다 (되=VV) — PDF §2: 의미 없음
    ("게", "EC", "하", "VX"): (None, "Auxiliary Verb"),                           # -게 하다 (하=VX 보조용언)
    (ANY, "ETM", "것", "NNB", "같", ANY): ("guess, it might be", "Auxiliary Verb"),
    (ANY, "ETM", "거", "NNB", "같", ANY): ("guess, it might be", "Auxiliary Verb"),
    ("으라", "EF", "니", "EF"): ("asking back / showing surprise", "Sentence end"),
    ("시", ANY, "ᆸ시오", "EF"): ("Formal Command", "Sentence end"),
    ("시", "EP", "ᆸ시오", "EF"): ("Formal Command", "Sentence end"),
    ("으시", "EP", "ᆸ시오", "EF"): ("Formal Command", "Sentence end"),
    ("으시", ANY, "ᆸ시오", "EF"): ("Formal Command", "Sentence end"),
    ("고", "EC", "말", "VX", "고", "EF"): ("of course", "Sentence end"),
    ("ᆫ다", "EF", "던가", "EF"): ("Hearsay Question", "Sentence end"),
    ("는다", "EF", "던가", "EF"): ("Hearsay Question", "Sentence end"),
    ("ᆫ다", "EF", "던가", "JX"): ("Hearsay Question", "Sentence end"),   # 분석기 변이형
    ("는다", "EF", "던가", "JX"): ("Hearsay Question", "Sentence end"),
    ("ᆫ", "EF", "다던데", "EF"): ("Hearsay Remark", "Sentence end"),
    ("는", "EF", "다던데", "EF"): ("Hearsay Remark", "Sentence end"),
    # 종결어미 compound 추가 (엑셀 0410 태깅 목록)
    ("는", "ETM", "다면서", "EF"): ("Hearsay Question", "Sentence end"),
    ("는", "ETM", "다니", "EF"): ("asking back / showing surprise", "Sentence end"),
    ("달", "VX", "오", ANY): ("please / give me / Soft Statement", "Sentence end"),
    ("을", "ETM", "라", "EF"): ("I'm afraid~", "Sentence end"),
    ("려", "EC", "ᆸ니까", "EF"): ("Would you~?", "Sentence end"),
    ("려", "EC", "ᆸ니까", ANY): ("Would you~?", "Sentence end"),
    ("려", "EC", "이", "VCP", "ᆸ니다", "EF"): ("I would like to~", "Sentence end"),
    ("으려", "EC", "이", "VCP", "ᆸ니다", "EF"): ("I would like to~", "Sentence end"),
    ("으라면", "EC", "서", "NNG"): ("You said that~", "Sentence end"),
    ("다", "EC", "이", "VCP", "오", "EF"): ("please / give me / Soft Statement", "Sentence end"),
    ("던", "ETM", "데", "NNB"): ("recollection", "Connector"),
    ("을", "ETM", "것", "NNB", "이", "JKS"): ("promise ~ like futures", "Particle"),
    ("을", "ETM", "거", "NNB", "ᆯ", ANY): ("I guess / I should have", "Sentence end"),
    ("은", "ETM", "걸", "NNB", "ㄹ", "JKO"): ("I guess / Actually", "Sentence end"),
    ("ᆫ", "ETM", "걸", "NNB", "ㄹ", "JKO"): ("I guess / Actually", "Sentence end"),
    ("은", "ETM", "걸", "NNB", "ᆯ", "JKO"): ("I guess / Actually", "Sentence end"),
    ("ᆫ", "ETM", "걸", "NNB", "ᆯ", "JKO"): ("I guess / Actually", "Sentence end"),
    ("자라", "VV", "니까", "EC"): ("Emphatic statement", "Sentence end"),
    ("으려", ANY, "ᆸ니다", ANY): ("I would like to~", "Sentence end"),
    ("으라", "EF", "면서", "EF"): ("You said that~", "Sentence end"),
    ("으라면", ANY, "서", ANY): ("You said that~", "Sentence end"),
    ("ᆫ", "ETM", "걸", "NNB", "ᆯ", ANY): ("I guess / Actually", "Sentence end"),
    ("은", "ETM", "걸", "NNB", "ᆯ", ANY): ("I guess / Actually", "Sentence end"),
    ("ᆫ", "ETM", "거", "NNB", "ᆯ", ANY): ("I guess / Actually", "Sentence end"),
    ("은", "ETM", "거", "NNB", "ᆯ", ANY): ("I guess / Actually", "Sentence end"),
    # 연결어미 compound 추가 (엑셀 0410 태깅 목록)
    ("어야", "EC", "하", "VV", "지", "EC"): ("only if", "Connector"),
    ("자", "EC", "길래", "EC"): ("because one suggested to", "Connector"),
    # 엑셀에서 분석기 오류로 나타나는 복합 분석형 (참고용 — 실제 매칭은 드뭄)
    ("는다기", "ETN", "에", "JKB"): ("I heard that ... so ...", "Connector"),
    ("어다", "EC", "가", "JKS"): ("and", "Connector"),
    # ④ Negation Pattern (PDF 형태소 표시 규칙 §4) — 의미=Negation, 품사 없음
    ("지", "EC", "않", "VX"): ("Negation", None),                         # -지 + 않
    ("지", "EC", "말", "VX"): ("Negation", None),                         # -지 + 말
    # ② 보조용언 중 부정: -지 + 못(하) — 의미=cannot, 품사=Auxiliary Verb (PDF §2)
    ("지", "EC", "못하", "VX"): ("cannot", "Auxiliary Verb"),              # -지 + 못하 (단일 토큰)
    ("지", "EC", "못", "MAG", "하", "XSV"): ("cannot", "Auxiliary Verb"), # -지 + 못(MAG)+하(XSV) — Kiwi 분리형
    # ⑤ Adverbial Form Pattern (PDF 형태소 표시 규칙 §5 추가 예외)
    ("에", "JKB", "따르", "VV", "면", "EC"): ("According to", "Adverbial Form"),
    # 으로 인하여: Kiwi 분석 → 인(NNG)+하(XSV)+어/여(EC) (인하여가 인+하+어/여로 분리됨)
    ("으로", "JKB", "인", "NNG", "하", "XSV", "어", "EC"): ("due to", "Adverbial Form"),
    ("으로", "JKB", "인", "NNG", "하", "XSV", "여", "EC"): ("due to", "Adverbial Form"),
    ("으로", "JKB", "인하", "VV", "어", "EC"): ("due to", "Adverbial Form"),   # 단일 동사형 변이
    ("으로", "JKB", "인하", "VV", "여", "EC"): ("due to", "Adverbial Form"),   # 단일 동사형 변이
    # 받침 없는 명사 + 로(JKB): 실수로 인해, 사고로 인하여 등
    ("로", "JKB", "인", "NNG", "하", "XSV", "어", "EC"): ("due to", "Adverbial Form"),
    ("로", "JKB", "인", "NNG", "하", "XSV", "여", "EC"): ("due to", "Adverbial Form"),
    ("로", "JKB", "인하", "VV", "어", "EC"): ("due to", "Adverbial Form"),
    ("로", "JKB", "인하", "VV", "여", "EC"): ("due to", "Adverbial Form"),
    # 기 위하여: Kiwi 분석 → 위하(VV)+어/여/여서(EC)
    ("기", "ETN", "위하", "VV", "어", "EC"): ("in order to", "Adverbial Form"),
    ("기", "ETN", "위하", "VV", "여", "EC"): ("in order to", "Adverbial Form"),
    ("기", "ETN", "위하", "VV", "여서", "EC"): ("in order to", "Adverbial Form"),
    ("기", "ETN", "위하", "VV", "어서", "EC"): ("in order to", "Adverbial Form"),
}

# ============================================================================
# 접사 결합 규칙 (Affix Compound Rules)
# 정책: 접두사·접미사·파생 접사는 어근과 붙여서 하나의 단위로 처리
# 형태소: (form1, pos1, form2, pos2) → (합쳐진_글로스, pos_eng)
# 글로스는 None으로 두고, glossing.py에서 런타임에 구성 (사전/규칙 적용 후 합침)
# ============================================================================
AFFIX_COMPOUND = {
    # 동사 파생: NNG + XSV (하) → Action Verb
    # 예: 공부+하, 회의+하, 노력+하, 운동+하, 준비+하, 존중+하
    (ANY, "NNG", "하", "XSV"): (None, "Action Verb"),  # 글로스는 None (런타임 구성)
    (ANY, "NNG", "하", "XSA"): (None, "Action Verb"),  # 회의하(XSA 변이)

    # 형용사 파생: NNG + XSA (-스럽다, -롭다 등) → Descriptive Verb
    # 예: 사랑+스럽, 신뢰+롭, 도움+이+되, 부담+스럽
    (ANY, "NNG", ANY, "XSA"): (None, "Descriptive Verb"),

    # 형용사 파생: XR + XSA (부사→형용사 변환) → Descriptive Verb
    # 예: 똑똑+하
    (ANY, "XR", ANY, "XSA"): (None, "Descriptive Verb"),

    # 명사 파생: NNG + XSN (-들, -님 등) → Noun
    # 예: 아이+들, 선생+님, 학생+들
    (ANY, "NNG", ANY, "XSN"): (None, "Noun"),

    # 부사 파생: XR + XSM (부사 파생) → Adverb
    # 예: 조용+히(XSM으로 분석되는 경우)
    (ANY, "XR", ANY, "XSM"): (None, "Adverb"),
}


class SpanLabels:
    """스팬 레이블 내부 식별자 — PDF(한국어_문장_분석_구절_정리) 공식 태그명 기준

    - 이 값은 코드 내부 식별자로만 사용
    - 화면 표시 문자열은 SPAN_DISPLAY dict에서 별도 관리
    - 절 내부 위치(주절/종속절)와 무관하게 구는 단일 레이블 사용
    """

    # 절 (Clause Layer)
    SENT      = "Sentence"
    MAINC     = "MainC"
    SUBC      = "SubC"
    QUOTEC_DIR = "QuoteC_Dir"
    QUOTEC_IND = "QuoteC_Ind"
    EMC_ADJ   = "EmC_Adj"   # 관형절
    EMC_N     = "EmC_N"     # 명사절
    EMC_ADV   = "EmC_Adv"   # 부사절

    # 구 (Phrase Layer)
    VP   = "VP"
    ADJP = "AdjP"
    ADVP = "AdvP"
    NP   = "NP"
    NPS  = "NPS"
    SP   = "SP"   # Subject Phrase
    OP   = "OP"   # Object Phrase
    CP   = "CP"   # Complement Phrase
    TP   = "TP"   # Topic Phrase


