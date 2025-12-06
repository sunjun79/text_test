# app.py  (C:\Law_Chatbot 에 저장)

import streamlit as st

from Chatbot import (
    train_classifier_if_needed,
    load_risk_table,
    split_contract_text,
    explain_clause,
    get_checklist_and_tips,
    RISK_CSV_PATH,
)

# ================================
# 1. 모델 + 위험도 테이블 로딩
# ================================
@st.cache_resource
def load_model_and_risk():
    model = train_classifier_if_needed()      # 이미 학습된 모델 있으면 불러오고, 없으면 학습
    risk_map = load_risk_table(RISK_CSV_PATH)
    return model, risk_map


model, risk_map = load_model_and_risk()

# 이모지 매핑
RISK_EMOJI = {
    "H": "🔴 위험",
    "M": "🟡 주의",
    "L": "🟢 안전",
}

RISK_ORDER = {"H": 0, "M": 1, "L": 2}


# ================================
# 2. Streamlit UI
# ================================
st.set_page_config(page_title="계약서 도우미 챗봇", layout="wide")

st.title("📄 LLM 기반 계약서 도우미 (웹 버전)")
st.write(
    """
계약서 내용을 입력하면,

1. **조항별 라벨(조항 종류)** 를 예측하고  
2. **위험도(H/M/L)** 를 매긴 뒤  
3. 선택한 조항에 대해  
   - 위험/주의 이유 설명  
   - 체크리스트  
   - 협상 가이드  
를 보여줍니다.

> 한 줄에 한 조항씩 넣으면 가장 깔끔하게 동작합니다.
"""
)

st.markdown("---")

# 왼쪽: 입력 / 옵션, 오른쪽: 결과
col1, col2 = st.columns([1.2, 1.8])

with col1:
    st.subheader("1️⃣ 계약서 입력")

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
        help="조항별로 줄바꿈(Enter)을 넣어주면 더 정확하게 분석됩니다.",
    )

    top_n = st.slider(
        "위험/주의 조항 몇 개까지 볼까요?",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
    )

    analyze_btn = st.button("🔍 계약서 분석하기")

with col2:
    st.subheader("2️⃣ 분석 결과")

    if analyze_btn:
        if not contract_text.strip():
            st.warning("먼저 계약서 내용을 입력해 주세요.")
        else:
            # --------------------------
            # (1) 조항 분리
            # --------------------------
            clauses = split_contract_text(contract_text)
            if not clauses:
                st.warning("분석할 문장이 없습니다. 줄바꿈을 포함해서 다시 입력해 주세요.")
            else:
                st.info(f"총 **{len(clauses)}개**의 조항(문장)을 분석했습니다.")

                # --------------------------
                # (2) 예측
                # --------------------------
                preds = model.predict(clauses)

                results = []
                for idx, (clause, label) in enumerate(zip(clauses, preds), start=1):
                    label = str(label)
                    risk = risk_map.get(label, "M")
                    results.append(
                        {
                            "idx": idx,      # 원래 문장 번호
                            "clause": clause,
                            "label": label,
                            "risk": risk,
                        }
                    )

                # 위험도 순으로 정렬 (H → M → L)
                results.sort(key=lambda x: RISK_ORDER.get(x["risk"], 1))

                # 상위 N개만 사용
                top_results = results[:top_n]

                # --------------------------
                # (3) 요약 리스트 출력
                # --------------------------
                st.markdown("#### 🔎 위험/주의 조항 요약")

                if not top_results:
                    st.write("표시할 조항이 없습니다.")
                else:
                    for i, item in enumerate(top_results, start=1):
                        risk_str = RISK_EMOJI.get(item["risk"], item["risk"])
                        st.markdown(
                            f"""
**[{i}] {risk_str}**  
- 원문 조항 번호: `{item['idx']}`  
- 라벨: `{item['label']}`  
- 조항 내용:  
> {item['clause']}
"""
                        )

                    # --------------------------
                    # (4) 상세 설명용 선택 박스
                    # --------------------------
                    st.markdown("---")
                    st.markdown("#### 📝 상세 설명 듣기")

                    option_indices = list(range(len(top_results)))
                    option_labels = [
                        f"[{i+1}] {RISK_EMOJI.get(item['risk'], item['risk'])} / 라벨: {item['label']}"
                        for i, item in enumerate(top_results)
                    ]

                    selected = st.selectbox(
                        "상세 설명이 궁금한 조항을 선택하세요.",
                        options=option_indices,
                        format_func=lambda i: option_labels[i],
                    )

                    selected_item = top_results[selected]
                    clause = selected_item["clause"]
                    label = selected_item["label"]
                    risk = selected_item["risk"]

                    # --------------------------
                    # (5) 상세 설명 / 체크리스트 / 협상 가이드
                    # --------------------------
                    st.markdown("##### 📌 선택한 조항 원문")
                    st.info(clause)

                    explanation = explain_clause(label, risk, clause)
                    checklist, tips = get_checklist_and_tips(label, risk)

                    st.markdown("##### 📚 근거·해설")
                    st.write(explanation)

                    st.markdown("##### ✅ 체크리스트 (계약 체결 전 확인할 사항)")
                    if checklist:
                        for c in checklist:
                            st.markdown(f"- {c}")
                    else:
                        st.write("체크리스트가 정의되어 있지 않은 조항입니다.")

                    st.markdown("##### 🤝 협상 가이드 (상대방과 조정할 포인트)")
                    if tips:
                        for t in tips:
                            st.markdown(f"- {t}")
                    else:
                        st.write("협상 가이드가 별도로 정의되어 있지 않은 조항입니다.")

    else:
        st.info("왼쪽에 계약서를 입력하고 **[🔍 계약서 분석하기]** 버튼을 눌러주세요.")
