# chat_app.py  (C:\Law_Chatbot 에 저장)
# Streamlit 기반 "챗봇형" 계약서 도우미 + 경량 LLM(Ollama) 통합 버전
# - 계약서 입력/분석: 왼쪽 사이드바 (다크 모드 스타일)
# - 메인 화면: 상단 그라데이션 헤더 + 하단 Kakao 스타일 채팅 UI

import re
import streamlit as st

from Chatbot import (
    train_classifier_if_needed,
    load_risk_table,
    explain_clause,
    get_checklist_and_tips,
    RISK_CSV_PATH,
    analyze_contract_to_dict,
)
from chat import call_local_llm  # 네가 만든 chat.py 의 함수
from rag_utils import load_or_build_chroma_db, rag_search


# ================================
# 0. 페이지 설정 / 상수
# ================================
st.set_page_config(
    page_title="LLM 계약서 도우미 (챗봇)",
    page_icon="📄",
    layout="wide",
)

RISK_EMOJI = {
    "H": "🔴 위험",
    "M": "🟡 주의",
    "L": "🟢 안전",
}
RISK_ORDER = {"H": 0, "M": 1, "L": 2}

# LLM 사용 여부 (느리면 False 로 꺼도 됨)
USE_LLM = True
LLM_MODEL_NAME = "gemma2:2b"  # chat.py 기본값도 이걸로 맞춰두면 편함


# ================================
# 1. CSS 주입 (사이드바 + 헤더 + 채팅 말풍선)
# ================================
def inject_chat_css():
    st.markdown(
        """
        <style>
        /* ===== 전체 배경 / 기본 컨테이너 ===== */
        [data-testid="stAppViewContainer"] {
            background: #f3f4f6;
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1.5rem;
            background: transparent;
        }

        /* ===== 사이드바 : 다크모드 + 흰 글씨 ===== */
        [data-testid="stSidebar"] {
            background: #020617;
        }
        [data-testid="stSidebar"] * {
            color: #f9fafb !important;
            
        }
        [data-testid="stSidebar"] textarea {
            background: #020617;
            color: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
        }
        [data-testid="stSidebar"] .stButton button {
            background: #6366f1;
            color: #f9fafb;
            border: none;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.95rem;
            padding-top: 0.6rem;
            padding-bottom: 0.6rem;
        }

        /* ===== 상단 그라데이션 헤더 (메인 영역) ===== */
        .hero-gradient {
            width: 100%;
            border-radius: 24px;
            background: linear-gradient(135deg, #4f46e5, #6366f1, #ec4899);
            color: #f9fafb;
            padding: 28px 40px 30px 40px;
            box-sizing: border-box;
            margin: 0 auto 18px auto;
            font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
        }
        .hero-title {
            font-size: 1.7rem;
            font-weight: 800;
            margin-bottom: 10px;
        }
        .hero-desc {
            font-size: 0.98rem;
            line-height: 1.6;
            opacity: 0.97;
        }
        .hero-warning {
            margin-top: 12px;
            font-size: 0.9rem;
            line-height: 1.5;
            opacity: 0.95;
        }

        /* ===== 채팅 영역 ===== */
        .chat-wrapper {
            padding: 4px 4px 4px 4px;
            min-height: 260px;
        }

        .chat-row {
            display: flex;
            margin-bottom: 6px;
        }
        .chat-row.assistant-row {
            justify-content: flex-start;
        }
        .chat-row.user-row {
            justify-content: flex-end;
        }

        .chat-bubble {
            max-width: 80%;
            padding: 8px 12px;
            font-size: 0.9rem;
            line-height: 1.45;
            border-radius: 16px;
            word-break: break-word;
            white-space: pre-wrap;
            font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
        }

        /* 어시스턴트 말풍선 (왼쪽, 연한 회색) */
        .assistant-bubble {
            background: #f9fafb;
            color: #111827;
            border-radius: 16px 16px 16px 4px;
            border: 1px solid #e5e7eb;
        }

        /* 유저 말풍선 (오른쪽, 파랑) */
        .user-bubble {
            background: #2563eb;
            color: #f9fafb;
            border-radius: 16px 16px 4px 16px;
        }

        /* ===== chat_input (하단 입력창) ===== */
        .stChatInputContainer {
        padding: 0;
        }
        
        .stChatInput textarea {
        background: #f9fafb !important;
        border-radius: 999px !important;
        border: 1px solid #d1d5db !important;
        color: #111827 !important;
        
        /* ▼▼ 높이/패딩 문제 해결 ▼▼ */
        padding: 0.6rem 1rem !important;   /* 입력창 내부 여백 늘림 */
        min-height: 2.8rem !important;     /* 최소 높이 확보 */
        line-height: 1.4 !important;       /* 텍스트 줄 높이 조정 */
        font-size: 0.95rem !important;     /* 글자 크기 */
        resize: none !important;           /* 크기조절 핸들 제거 (선택) */
        }
        
        .stChatInput textarea::placeholder {
        color: #9ca3af !important;
        }
        
        /* ========= Expander 아이콘 텍스트 숨기기 ========= */
        [data-testid="stSidebar"] [data-baseweb="icon"] {
            font-size: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ================================
# 2. 모델 + 위험도 테이블 로딩
# ================================
@st.cache_resource
def load_model_and_risk():
    model = train_classifier_if_needed()
    risk_map = load_risk_table(RISK_CSV_PATH)
    return model, risk_map


@st.cache_resource
def load_rag_db():
    """
    법령·표준약관 PDF들을 임베딩해 둔 Chroma DB 로더.
    한 번 로딩하면 Streamlit이 캐시해 둠.
    """
    return load_or_build_chroma_db()


# 전역에서 한 번만 로딩
model, risk_map = load_model_and_risk()
rag_db = load_rag_db()


# ================================
# 3. LLM 리라이팅 훅 (Ollama)
# ================================
def rephrase_with_llm(text: str) -> str:
    """
    규칙 기반으로 만든 답변(text)을
    로컬 LLM(Ollama)을 이용해서 조금 더 자연스럽게 다듬는다.

    - 내용/구조는 유지
    - 단정적 법률 자문은 하지 않도록 프롬프트로 제어
    - Ollama 에러나면 원문 그대로 반환
    """
    if not USE_LLM:
        return text

    prompt = f"""
너는 한국어로 말하는 '계약서 설명 보조 도우미'야.

아래 텍스트는 이미 규칙 기반 시스템이 생성한
계약서 조항 설명/체크리스트/협상 가이드야.

이 텍스트를 다음 기준에 맞게 '표현만' 자연스럽게 다듬어줘:
- 내용의 사실 관계 및 구조(항목, 번호, 리스트)는 유지
- 이미 포함된 면책 문구/주의 문구는 반드시 유지
- "이렇게 하면 무조건 된다" 같은 단정적인 법률 자문으로 바꾸지 마
- 존댓말로 차분하게 작성

[원문 텍스트 시작]
{text}
[원문 텍스트 끝]
"""

    try:
        llm_out = call_local_llm(prompt, model=LLM_MODEL_NAME)
        llm_out = (llm_out or "").strip()
        return llm_out if llm_out else text
    except Exception as e:
        # Ollama 가 꺼져 있거나 오류가 나면 그냥 원문 반환
        print("LLM 호출 오류:", e)
        return text
    
def build_explanation_with_rag(clause: str, label: str, risk: str) -> str:
    """
    단일 조항에 대해:
    - 기존 규칙 기반 설명 + 체크리스트 + 협상 가이드 생성
    - RAG로 관련 법령/표준약관 요약 가져오기
    - LLM에게 이걸 한 번에 정리해 달라고 요청

    최종적으로 '이 조항 상세 설명' 본문 텍스트를 반환.
    """
    # 1) 규칙 기반 초안
    base_expl = explain_clause(label, risk, clause)
    checklist, tips = get_checklist_and_tips(label, risk)

    checklist_lines = []
    if checklist:
        for c in checklist:
            checklist_lines.append(f"- {c}")
    else:
        checklist_lines.append("- 별도로 정의된 체크리스트는 없습니다.")

    tips_lines = []
    if tips:
        for t in tips:
            tips_lines.append(f"- {t}")
    else:
        tips_lines.append("- 별도로 정의된 협상 가이드는 없습니다.")

    # 2) RAG로 관련 법령/표준약관 컨텍스트 검색
    query = f"{label} {clause}"
    try:
        contexts = rag_search(query, rag_db, top_k=3)
    except Exception as e:
        print("RAG 검색 오류(조항 상세 설명):", e)
        contexts = []

    if contexts:
        ctx_block = "\n\n".join(
            [f"[참고 {i+1}] {ctx}" for i, ctx in enumerate(contexts)]
        )
    else:
        ctx_block = (
            "(관련 법령·표준약관에서 직접 연결되는 조문을 찾지 못했습니다. "
            "일반적인 계약 관행을 기준으로 설명합니다.)"
        )

    # 3) LLM에게 '초안 + RAG 컨텍스트'를 가지고 정리해달라고 요청
    draft = f"""
[사용자 계약서 조항]
{clause}

[ML 분류 결과]
- 라벨: {label}
- 위험도: {risk}

[규칙 기반 설명 초안]
{base_expl}

[체크리스트 초안]
{chr(10).join(checklist_lines)}

[협상 가이드 초안]
{chr(10).join(tips_lines)}

[참고용 법령·표준약관 요약 (RAG)]
{ctx_block}
"""

    prompt = f"""
너는 한국어로 답변하는 '계약서 설명 도우미'야.

아래 DRAFT는
1) ML/규칙 기반으로 만든 조항 설명/체크리스트/협상가이드,
2) 법령·표준약관에서 RAG로 찾아온 참고 내용을 합쳐 놓은 초안이다.

[DRAFT]
{draft}
[DRAFT 끝]

위 정보를 바탕으로, 사용자가 이해하기 쉽게 다음 구조로 정리해줘:

1. **근거·해설**
   - 이 조항이 어떤 내용인지
   - 왜 사용자에게 위험/주의가 될 수 있는지
   - (가능하면) 위에 제시된 참고 법령·표준약관과 비교해서, 어떤 점이 일반적인 수준보다 불리해 보일 수 있는지 서술
2. **체크리스트 (계약 체결 전 확인할 것들)**
   - 사용자가 실제로 확인해야 할 포인트를 불릿 리스트로 정리
3. **협상 가이드 (상대방과 조정할 포인트)**
   - 어떻게 조정 요청을 해볼 수 있는지 제안
4. **면책 문구**
   - 이 설명은 일반적인 정보 제공일 뿐이고, 실제 분쟁/소송/계약 체결은 변호사와 상담해야 한다는 내용을 반드시 포함

주의사항:
- DRAFT에 있는 사실 관계/취지는 유지하면서, 표현만 더 자연스럽게 다듬어.
- '무조건 위법하다', '반드시 승소한다'처럼 단정적인 법률 판단은 하지 마.
- 존댓말로 차분하게 작성해.
"""

    try:
        llm_out = call_local_llm(prompt, model=LLM_MODEL_NAME)
        llm_out = (llm_out or "").strip()
        return llm_out if llm_out else draft
    except Exception as e:
        print("LLM 오류(build_explanation_with_rag):", e)
        # LLM이 터지면 최소한 draft라도 그대로 보여주자
        return draft


# ================================
# 4. 세션 상태 초기화
# ================================
def init_session_state():
    if "analysis_results" not in st.session_state:
        st.session_state["analysis_results"] = []  # 조항 분석 결과 리스트
    if "messages" not in st.session_state:
        # role: "user" / "assistant"
        st.session_state["messages"] = []
    if "contract_loaded" not in st.session_state:
        st.session_state["contract_loaded"] = False


# ================================
# 5. 규칙 기반 질의 응답 함수
# ================================

def detect_intent(q: str) -> str:
    """
    사용자의 질문(q)을 보고, 대략적인 '질문 종류(intent)'를 문자열로 반환한다.
    """
    # 공백 제거 + 소문자 버전 (일부 키워드용)
    q_no_space = q.replace(" ", "").lower()

    # 1) 시스템 동작 관련 질문 (정렬 기준, 순서 등)
    if ("정렬" in q) or ("순서" in q) or ("어떻게 정리" in q):
        return "system_info"

    # 2) 위험한 법률 자문 / 변호사 추천 관련
    #    - 변호사, 로펌, 법률사무소 + 추천/소개/연결
    if any(k in q for k in ["변호사", "로펌", "법률사무소"]) and any(
        k in q for k in ["추천", "소개", "연락", "연결", "알려줘"]
    ):
        return "legal_advice_risky"

    #    - 소송하면 이길지, 서명해도 되는지 등
    risky_phrases = [
        "써명해도돼",
        "서명해도돼",
        "소송하면이겨",
        "승소가능",
        "이길까",
        "해도될까",
        "소송해도될까",
    ]
    if any(k in q_no_space for k in risky_phrases):
        return "legal_advice_risky"

    # 3) 특정 조항 번호 질문 (N번 조항)
    if re.search(r"\d+\s*번", q):
        return "clause_detail"

    # 4) 가장 위험한 / 위험 요약 질문
    if any(k in q for k in ["가장 위험", "제일 위험", "위험한 조항"]):
        return "risk_summary"

    # 5) 조항 유형(키워드) 기반 질문 (위약금, 해지, 손해배상 등)
    key_words = ["위약금", "손해배상", "해지", "해제", "관할법원", "분쟁", "보증", "담보", "비밀유지"]
    if any(k in q for k in key_words):
        return "label_explanation"

    # 6) 전역 요약: "이 계약서에서 어디를 조심해야 해?" 같은 질문
    if any(
        k in q
        for k in [
            "전체적으로",
            "전반적으로",
            "특히 봐야",
            "어디를 봐야",
            "어디를 조심",
            "무엇을 조심",
            "어디가 위험",
        ]
    ):
        return "global_summary"

    # 7) 계약/계약서에 대한 일반 질문 (계약서란 무엇인가 등)
    if ("계약서" in q) or ("계약" in q):
        return "general_contract"

    # 8) 나머지는 이 도구 범위를 벗어난(out_of_scope) 질문으로 처리
    return "out_of_scope"


def answer_legal_advice_risky(q: str) -> str:
    """
    소송 승소 가능성, 서명해도 되는지, 특정 변호사 추천 등
    '법률 자문' 영역에 해당하는 질문에 대한 공통 답변.
    """
    base = (
        "이 도구는 개별 사건에 대해 **서명 여부, 소송 승소 가능성, 특정 변호사 추천**과 같은 "
        "법률 자문을 제공하지 않도록 설계되어 있습니다.\n\n"
        "제가 도와드릴 수 있는 범위는, 계약서 조항의 구조와 일반적인 위험 포인트를 "
        "설명해 드리는 정도입니다. 실제 분쟁, 소송, 변호사 선임이 필요한 상황이라면 "
        "반드시 변호사 등 법률 전문가와 직접 상담하시길 권장드립니다."
    )
    return rephrase_with_llm(base)


def answer_out_of_scope(q: str) -> str:
    """
    계약서와 거의 관련 없는 질문에 대한 공통 답변.
    """
    base = (
        "현재 이 챗봇은 **계약서 내용 분석 전용 도구**로 만들어져 있습니다.\n\n"
        "지금 질문은 계약서 조항이나 위험도 설명과는 직접적인 관련이 없어서, "
        "정확한 답변을 드리기 어렵습니다.\n\n"
        "계약서 안의 특정 문장, 이해가 안 되는 조항, 위험해 보이는 부분을 중심으로 "
        "다시 질문해 주시면 그 범위 안에서 최대한 도와드릴게요."
    )
    return rephrase_with_llm(base)


def answer_system_info(q: str) -> str:
    """
    '정렬 기준이 뭐야?' 같이 시스템 동작 방식을 묻는 질문에 대한 답변.
    (지금 구조를 그대로 설명해 주면 됨)
    """
    base = (
        "지금 화면에 보이는 조항 목록은 기본적으로 **위험도 순(H → M → L)**으로 먼저 정렬한 다음,\n"
        "같은 위험도 안에서는 원래 조항 번호(idx) 순으로 정렬하고 있습니다.\n\n"
        "이 기준은 코드 안에 고정값(RISK_ORDER)으로 들어 있어서, "
        "나중에 필요하면 '원래 계약서 순서대로', '날짜순' 같은 다른 방식으로도 바꿀 수 있어요."
    )
    return rephrase_with_llm(base)

def answer_user_question(question: str) -> str:
    """
    사용자의 질문(question)과 세션에 저장된 분석 결과를 기반으로
    규칙 + (옵션) LLM을 사용해 답변을 만든다.
    이제는 '질문 intent'를 먼저 판별해서 안전하게 처리한다.
    """
    q = question.strip()

    # 0) 질문 의도(intent) 먼저 판별
    intent = detect_intent(q)
    
    # 1) 법률 자문/변호사 추천 등 '위험한 질문'은 바로 차단
    if intent == "legal_advice_risky":
        return answer_legal_advice_risky(q)

    # 2) 시스템 동작 관련 질문 (정렬 기준 등)
    if intent == "system_info":
        return answer_system_info(q)

    # 2.5) 법령·표준약관 / 표준계약 관련 질문이면 RAG로 우선 답변
    law_keywords = [
        "근로기준법", "민법", "상법",
        "표준근로계약서", "표준임대차계약서",
        "취업규칙", "저작권", "저작권 계약",
        "기술이전", "택배", "운송약관", "표준약관",
    ]

    if any(kw in q for kw in law_keywords):
        try:
            contexts = rag_search(q, rag_db, top_k=3)
            if not contexts:
                base = (
                    "지식 DB에서 직접적으로 일치하는 법령·표준약관 내용을 찾지는 못했습니다.\n"
                    "그래도 관련 키워드로 다시 질문해 보거나, 계약서 조항 기준으로 질문해 주세요."
                )
                return rephrase_with_llm(base)

            joined = "\n\n---\n\n".join(contexts)
            base = (
                "질문과 관련해서, 네가 구축한 **법령·표준계약/표준약관 지식 DB**에서\n"
                "다음과 같은 내용을 참고용으로 찾았습니다.\n\n"
                "⚠️ 아래 내용은 법률 자문이 아니라, 원문을 요약한 참고 정보입니다.\n"
                "실제 분쟁·해고·손해배상 등은 반드시 변호사와 상의해야 합니다.\n\n"
                f"{joined}"
            )
            return rephrase_with_llm(base)
        except Exception as e:
            print("RAG 검색 오류:", e)
            base = (
                "법령·표준약관 지식 DB 검색 중 오류가 발생했습니다.\n"
                "잠시 후 다시 시도해 보거나, 개별 조항 기준으로 질문해 주세요."
            )
            return rephrase_with_llm(base)

    # 3) 계약과 거의 관계 없는 질문 → out_of_scope
    if intent == "out_of_scope":
        return answer_out_of_scope(q)

    # 여기부터는 '계약서 분석 결과'가 필요함
    analysis_results = st.session_state.get("analysis_results", [])

    # 아직 계약서 분석 전이면 안내
    if not analysis_results:
        base = (
            "먼저 왼쪽 사이드바에서 계약서를 입력하고 "
            "**[📄 계약서 분석 실행]** 버튼을 눌러 분석을 완료한 뒤에 질문해 주세요."
        )
        return rephrase_with_llm(base)

    # ---------- intent별 처리 ----------

    # A) 특정 조항 번호 질문: "3번 조항 설명해줘"
    if intent == "clause_detail":
        num_match = re.search(r"(\d+)\s*번", q)
        if not num_match:
            base = "어느 번호의 조항을 말씀하시는지 조금 더 구체적으로 알려주시면 좋겠습니다."
            return rephrase_with_llm(base)

        idx = int(num_match.group(1))
        target = None
        for item in analysis_results:
            if item["idx"] == idx:
                target = item
                break

        if target is None:
            base = f"{idx}번 조항을 찾을 수 없습니다. 번호가 올바른지 다시 한 번 확인해 주세요."
            return rephrase_with_llm(base)

        clause = target["clause"]
        label = target["label"]
        risk = target["risk"]

        # ✅ 여기서부터는 RAG + LLM 기반 상세 설명 사용
        body = build_explanation_with_rag(clause, label, risk)

        header_lines = []
        header_lines.append(f"**[{idx}번 조항 상세 설명]**")
        header_lines.append(f"- 라벨: `{label}` / 위험도: {RISK_EMOJI.get(risk, risk)}")
        header_lines.append("\n> " + clause)

        base = "\n".join(header_lines) + "\n\n" + body

        # build_explanation_with_rag 안에서 이미 LLM을 썼으므로,
        # 여기서는 rephrase_with_llm을 다시 호출하지 않는다.
        return base
    
    # B) 전체에서 가장 위험한 조항: "가장 위험한 조항 알려줘"
    if intent == "risk_summary":
        high = [r for r in analysis_results if r["risk"] == "H"]
        target = high if high else [r for r in analysis_results if r["risk"] == "M"]

        if not target:
            txt = "현재 분석된 계약서에서 특별히 '위험' 등급으로 분류된 조항은 없습니다."
            return rephrase_with_llm(txt)

        top_k = target[:3]
        lines = ["분석 결과, 다음 조항들이 상대적으로 **위험도가 높은 조항**으로 분류되었습니다:\n"]
        for item in top_k:
            lines.append(
                f"- **[{item['idx']}번 조항]** "
                f"(위험도: {RISK_EMOJI.get(item['risk'], item['risk'])}, "
                f"라벨: `{item['label']}`)\n  → {item['clause']}"
            )
        lines.append(
            "\n각 조항에 대해 더 자세한 설명이 필요하다면, "
            "`\"N번 조항 자세히 알려줘\"`와 같이 번호를 지정해서 물어봐 주세요."
        )
        base = "\n".join(lines)
        return rephrase_with_llm(base)

    # C) 특정 키워드(위약금, 해지 등)와 관련된 조항 찾기
    if intent == "label_explanation":
        key_words = ["위약금", "손해배상", "해지", "해제", "관할법원", "분쟁", "보증", "담보", "비밀유지"]
        hit_keywords = [kw for kw in key_words if kw in q]
        kw = hit_keywords[0] if hit_keywords else None

        if not kw:
            # 이 경우는 거의 없겠지만, 혹시를 위해 기본 요약으로 fallback
            intent = "global_summary"  # 아래에서 처리
        else:
            matched = [
                r
                for r in analysis_results
                if (kw in r["clause"]) or (kw in r["label"])
            ]
            if not matched:
                base = (
                    f"분석된 조항들 중에서 `{kw}`(이)가 직접적으로 나타나는 조항은 찾지 못했습니다.\n"
                    "다만, 유사한 위험 요소가 있을 수 있으니, 번호를 지정해서 개별 조항을 살펴보는 것도 좋습니다."
                )
                return rephrase_with_llm(base)

            top_k = matched[:5]
            lines = [
                f"`{kw}`(와)과 관련이 있어 보이는 조항들을 정리하면 다음과 같습니다:\n"
            ]
            for item in top_k:
                lines.append(
                    f"- **[{item['idx']}번 조항]** "
                    f"(위험도: {RISK_EMOJI.get(item['risk'], item['risk'])}, "
                    f"라벨: `{item['label']}`)\n  → {item['clause']}"
                )

            lines.append(
                "\n궁금한 조항이 있다면 `\"N번 조항 자세히 설명해줘\"`와 같이 번호를 지정해서 물어봐 주세요."
            )
            base = "\n".join(lines)
            return rephrase_with_llm(base)

    # D) 전역 요약: "이 계약서에서 특히 봐야 할 포인트 뭐야?"
    if intent == "global_summary":
        top = analysis_results[:5]
        lines = [
            "이 계약서에서 특히 주의해서 볼 필요가 있는 조항들을 간단히 정리해 드릴게요.\n"
        ]
        for item in top:
            lines.append(
                f"- **[{item['idx']}번 조항]** "
                f"(위험도: {RISK_EMOJI.get(item['risk'], item['risk'])}, "
                f"라벨: `{item['label']}`)\n  → {item['clause']}"
            )

        lines.append(
            "\n특정 조항이 궁금하다면 `\"3번 조항 설명해줘\"`, "
            "`\"위약금 관련해서 제일 위험한 조항 알려줘\"`처럼 질문해 보세요."
        )
        base = "\n".join(lines)
        return rephrase_with_llm(base)

    # E) 계약/계약서에 대한 일반 설명 (아주 간단한 버전)
    if intent == "general_contract":
        base = (
            "일반적으로 계약서는 당사자 사이의 권리와 의무를 글로 정리해 둔 문서입니다.\n\n"
            "이 도구는 그중에서도 특히, 사용자가 불리해질 수 있는 조항(위약금, 해지, 분쟁 해결 등)을 "
            "자동으로 찾아 표시하고, 주의할 점을 설명하는 데 초점을 맞추고 있습니다.\n\n"
            "구체적인 조항이 궁금하다면, '3번 조항 설명해줘', '가장 위험한 조항 알려줘'처럼 물어보시면 됩니다."
        )
        return rephrase_with_llm(base)

    # F) 그 밖의 애매한 질문은 '전역 요약' 형태로 fallback
    top = analysis_results[:5]
    lines = [
        "이 계약서에서 특히 주의해서 볼 필요가 있는 조항들을 간단히 정리해 드릴게요.\n"
    ]
    for item in top:
        lines.append(
            f"- **[{item['idx']}번 조항]** "
            f"(위험도: {RISK_EMOJI.get(item['risk'], item['risk'])}, "
            f"라벨: `{item['label']}`)\n  → {item['clause']}"
        )

    lines.append(
        "\n특정 조항이 궁금하다면 `\"3번 조항 설명해줘\"`, "
        "`\"위약금 관련해서 제일 위험한 조항 알려줘\"`처럼 질문해 보세요."
    )
    base = "\n".join(lines)
    return rephrase_with_llm(base)


# ================================
# 6. 메인 UI
# ================================
def main():
    init_session_state()
    inject_chat_css()

    # ----- 왼쪽 사이드바 : 계약서 입력 / 분석 -----
    with st.sidebar:
        st.header("📄 계약서 입력 및 분석")
        st.caption("계약서 내용을 붙여넣고 버튼을 눌러 조항별 위험도를 분석합니다.")

        # 예시 계약서 자동 입력
        default_text = (
            "본 계약은 2025년 1월 1일부터 2025년 12월 31일까지 유효하다.\n"
            "임차인은 목적물을 선량한 관리자의 주의의무를 다하여 사용하여야 한다.\n"
            "임차인의 귀책사유로 계약이 해지되는 경우, 임차인은 잔여 기간 동안의 임대료 전액을 위약금으로 지급한다.\n"
            "본 계약과 관련하여 분쟁이 발생하는 경우, 서울중앙지방법원을 전속적 합의 관할 법원으로 한다.\n"
        )

        contract_text = st.text_area(
            "계약서 내용을 붙여넣어 주세요.",
            value=default_text,
            height=260,
            help="한 줄에 한 조항씩 입력하면 더 정확하게 분석됩니다.",
        )

        if st.button("📄 계약서 분석 실행", use_container_width=True):
            if not contract_text.strip():
                st.warning("먼저 계약서 내용을 입력해 주세요.")
            else:
                with st.spinner("계약서를 분석하는 중입니다..."):
                    results = analyze_contract_to_dict(
                        model, risk_map, contract_text, sort_by_risk=True
                    )

                st.session_state["analysis_results"] = results
                st.session_state["contract_loaded"] = True

                if results:
                    msg = (
                        f"계약서 분석이 완료되었습니다. 총 **{len(results)}개**의 조항이 분석되었습니다.\n\n"
                        "예시 질문:\n"
                        "- `가장 위험한 조항 알려줘`\n"
                        "- `3번 조항 자세히 설명해줘`\n"
                        "- `위약금 관련해서 뭐가 위험한지 알려줘`"
                    )
                else:
                    msg = "분석할 수 있는 조항이 없습니다. 계약서 입력을 다시 확인해 주세요."

                st.session_state["messages"].append(
                    {"role": "assistant", "content": msg}
                )

        with st.expander("🔎 분석 상태", expanded=False):
            if st.session_state["analysis_results"]:
                st.success(
                    f"분석된 조항 수: {len(st.session_state['analysis_results'])}개\n"
                    "오른쪽 챗봇에서 질문해 보세요."
                )
            else:
                st.info("아직 분석된 계약서가 없습니다.")

    # ----- 오른쪽 메인 영역 : 상단 그라데이션 헤더 + 채팅 -----
    # 전체를 가운데 정렬 느낌으로
    _, center, _ = st.columns([0.02, 0.96, 0.02])
    with center:
        # 상단 그라데이션 헤더 (전체 상단 영역)
        st.markdown(
            """
            <div class="hero-gradient">
              <div class="hero-title">LLM 기반 계약서 도우미</div>
              <div class="hero-desc">
                이 서비스는 계약서 조항을 자동 분석하고,<br>
                위험도가 높을 수 있는 조항을 중심으로 설명·체크리스트·협상 가이드를 제공하는 도구입니다.
              </div>
              <div class="hero-warning">
                ⚠️ 이 서비스는 법률 자문이 아니며,<br>
                실제 계약 체결 또는 분쟁이 우려되는 경우 반드시 변호사 등 법률 전문가와 상의해야 합니다.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # st.markdown("---")  이 줄을 지우고 아래로 교체
        st.markdown(
            '<hr style="margin: 0.4rem 0 0.35rem 0;">',  # 위·아래 여백을 작게
            unsafe_allow_html=True,
            )



        for msg in st.session_state["messages"]:
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            content_html = content.replace("\n", "<br>")

            if role == "user":
                row_class = "chat-row user-row"
                bubble_class = "chat-bubble user-bubble"
            else:
                row_class = "chat-row assistant-row"
                bubble_class = "chat-bubble assistant-bubble"

            st.markdown(
                f"""
                <div class="{row_class}">
                  <div class="{bubble_class}">
                    {content_html}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ----- 맨 아래 입력창 (ChatGPT 스타일) -----
    user_input = st.chat_input(
        "계약서에 대해 궁금한 점을 자연어로 물어보세요. (예: '가장 위험한 조항이 뭐야?')"
    )

    if user_input:
        # 사용자 메시지 저장
        st.session_state["messages"].append(
            {"role": "user", "content": user_input}
        )

        # 규칙 기반 + (옵션) LLM 리라이팅
        answer = answer_user_question(user_input)

        # 어시스턴트 메시지 저장
        st.session_state["messages"].append(
            {"role": "assistant", "content": answer}
        )

        # 다시 렌더링
        st.rerun()


if __name__ == "__main__":
    main()

