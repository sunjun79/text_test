# Chatbot.py

import os
import json
import re
from typing import List, Dict, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
import joblib

# =====================================
# 0. 기본 경로 설정 (네 컴퓨터 기준)
# =====================================
BASE_DIR = r"C:\Law_Chatbot"

TRAIN_LABEL_PATH = os.path.join(BASE_DIR, "train_label.jsonl")
VALID_LABEL_PATH = os.path.join(BASE_DIR, "valid_label.jsonl")

# 학습된 모델 파일
MODEL_PATH = os.path.join(BASE_DIR, "contract_clause_clf.joblib")

# 라벨별 위험도 CSV
RISK_CSV_PATH = os.path.join(BASE_DIR, "contract_label_risk_levels.csv")


# =====================================
# 1. 데이터 로딩 / 전처리 함수들
# =====================================
def extract_text(obj: dict) -> str:
    """
    라벨 JSON 한 줄에서 텍스트만 뽑는 함수.
    주로 'text' 키를 쓰지만, 혹시 모를 경우를 대비해 후보 키 몇 개 확인.
    """
    candidate_keys = ["text", "content", "document", "본문", "clause_text"]
    for key in candidate_keys:
        if key in obj and isinstance(obj[key], str):
            return obj[key].strip()
    return ""


def load_labeled_data(path: str) -> Tuple[List[str], List[str]]:
    """
    jsonl 파일에서 (문장, 라벨) 리스트를 불러옴.
    라벨이 여러 개면 첫 번째만 사용.
    """
    texts = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = extract_text(obj)
            labs = obj.get("labels", [])
            if not text:
                continue
            if isinstance(labs, list) and len(labs) > 0:
                label = str(labs[0])
            elif isinstance(labs, str):
                label = labs
            else:
                continue

            texts.append(text)
            labels.append(label)

    return texts, labels


def train_classifier_if_needed() -> Pipeline:
    """
    모델 파일이 없으면 학습 후 저장,
    있으면 불러오기.
    (시간 단축을 위해 학습 데이터 일부만 사용)
    """
    if os.path.exists(MODEL_PATH):
        print(f"▶ 이미 학습된 모델이 있습니다. 불러옵니다: {MODEL_PATH}")
        return joblib.load(MODEL_PATH)

    print("▶ 학습된 모델이 없어 새로 학습을 시작합니다.")

    # 1) train / valid 데이터 로드
    print("  - train 데이터 로딩 중...")
    X_train, y_train = load_labeled_data(TRAIN_LABEL_PATH)
    print(f"    원래 train 샘플 수: {len(X_train)}")

    print("  - valid 데이터 로딩 중...")
    X_valid, y_valid = load_labeled_data(VALID_LABEL_PATH)
    print(f"    원래 valid 샘플 수: {len(X_valid)}")

    # =========================================
    # (A) 학습 시간 단축: 샘플 수 줄이기
    # =========================================
    MAX_TRAIN_SAMPLES = 50000   # 필요하면 50_000이나 30_000으로 더 줄여도 됨
    MAX_VALID_SAMPLES = 10000

    if len(X_train) > MAX_TRAIN_SAMPLES:
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        X_train = X_train[:MAX_TRAIN_SAMPLES]
        y_train = y_train[:MAX_TRAIN_SAMPLES]
        print(f"    → train 샘플 수를 {MAX_TRAIN_SAMPLES}개로 줄였습니다.")

    if len(X_valid) > MAX_VALID_SAMPLES:
        X_valid, y_valid = shuffle(X_valid, y_valid, random_state=42)
        X_valid = X_valid[:MAX_VALID_SAMPLES]
        y_valid = y_valid[:MAX_VALID_SAMPLES]
        print(f"    → valid 샘플 수를 {MAX_VALID_SAMPLES}개로 줄였습니다.")

    # =========================================
    # (B) TF-IDF/로지스틱 설정을 가볍게 + verbose
    # =========================================
    print("▶ 모델 학습 중...(TF-IDF + LogisticRegression, 경량 버전)")
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=30000,   # 50,000 → 30,000
                    ngram_range=(1, 1),   # (1,2) → (1,1) 단어만 사용
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=200,         # 1000 → 200
                    n_jobs=-1,
                    solver="saga",        # 큰 데이터에 적합 + verbose 지원
                    verbose=1,            # 학습 과정 로그 출력
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    print("▶ 검증 데이터 성능 평가:")
    y_pred = model.predict(X_valid)
    print(classification_report(y_valid, y_pred, digits=3))

    # 3) 모델 저장
    print(f"▶ 모델 저장: {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)
    print("완료!")

    return model


def load_risk_table(csv_path: str) -> Dict[str, str]:
    """
    contract_label_risk_levels.csv에서
    label -> risk_level(H/M/L) 딕셔너리를 생성
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"위험도 CSV를 찾을 수 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)
    mapping = {}
    for _, row in df.iterrows():
        label = str(row["label"]).strip()
        risk = str(row["risk_level"]).strip().upper()
        if risk not in ["H", "M", "L"]:
            risk = "M"
        mapping[label] = risk
    return mapping



def split_contract_text(text: str) -> List[str]:
    """
    계약서 텍스트를 '제 n 조' 단위로 분리하는 함수.

    - 기본 전략:
      1) '제숫자조' 패턴(예: 제1조, 제 2 조)을 기준으로만 split
      2) 각 조항 내부에 있는 줄바꿈/빈 줄은 그대로 유지
      3) 만약 '제 n 조' 패턴이 전혀 없으면, 이전처럼 '줄 단위'로 fallback

    예)
        제1조(계약 기간) ...
        제2조(목적물 사용 및 유지 관리) ...
    """
    if not text:
        return []

    # 윈도우/맥 개행 차이 정리
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []

    # 1) '제 n 조' 패턴 기준으로 분리
    #    (?=...) : 앞의 패턴을 소비하지 않고, 해당 위치를 split 기준으로 사용
    parts = re.split(r"(?=제\s*\d+\s*조)", normalized)

    # 앞뒤 공백 제거 + 빈 문자열 제거
    clauses = [p.strip() for p in parts if p.strip()]

    # 2) 제대로 '제 n 조'로 시작하는 조항들만 나왔는지 검사
    #    (계약서에 '제1조' 같은 게 없을 수도 있으니까)
    if clauses and all(re.match(r"^제\s*\d+\s*조", c) for c in clauses):
        return clauses

    # 3) 만약 '제 n 조' 패턴이 하나도 없거나, 형식이 이상하면
    #    이전 버전처럼 "줄 단위"로 fallback
    lines: List[str] = []
    for line in normalized.split("\n"):
        line = line.strip()
        if line:
            lines.append(line)
    return lines

# =====================================
# 2. 해설 / 체크리스트 / 협상 가이드
# =====================================
def explain_clause(label: str, risk: str, clause_text: str) -> str:
    """
    근거·해설 제시:
    - 라벨명과 위험도를 기반으로
    - 왜 이 조항이 사용자에게 위험/주의가 될 수 있는지 간단 설명
    - 항상 가드레일(면책 문구)을 포함
    """
    reason = ""

    if "손해배상" in label or "손해 배상" in label or "위약" in label:
        reason = (
            "이 조항은 손해배상이나 위약금과 관련된 내용으로 보입니다. "
            "위약금이 과도하거나, 실제 손해액과 무관하게 높은 금액을 정한 경우 "
            "사용자에게 매우 불리하게 작용할 수 있습니다. 계약 위반 상황에서 "
            "어느 정도 수준까지 부담해야 하는지, 법에서 인정하는 범위인지 확인할 필요가 있습니다."
        )
    elif "해제" in label or "해지" in label or "종료" in label:
        reason = (
            "이 조항은 계약 해제·해지·종료 조건과 관련된 내용으로 보입니다. "
            "상대방에게만 해지 권한이 있거나, 사용자가 해지할 수 있는 조건이 "
            "지나치게 제한되어 있으면 일방적으로 불리한 계약이 될 수 있습니다."
        )
    elif "관할법원" in label or "중재" in label or "분쟁" in label:
        reason = (
            "이 조항은 분쟁 발생 시 어느 법원(또는 중재기관)을 이용할지 정하는 내용으로 보입니다. "
            "상대방이 있는 지역의 법원을 관할로 정해 두면, 사용자가 소송을 제기하거나 "
            "방어하는 과정에서 시간·비용 부담이 커질 수 있습니다."
        )
    elif "담보" in label or "보증" in label:
        reason = (
            "이 조항은 담보나 보증과 관련된 내용으로 보입니다. "
            "사용자가 제공하는 담보의 범위나 보증 책임의 범위가 넓게 설정된 경우, "
            "예상보다 큰 금전적 부담을 지게 될 수 있습니다."
        )
    elif risk == "H":
        reason = (
            "이 조항은 전반적으로 금전적 책임, 계약 해지, 강제 집행 등과 연결될 가능성이 높은 내용입니다. "
            "표준 약관이나 공정거래위원회에서 제시하는 조항들과 비교해 "
            "과도하게 한쪽에게 불리하지 않은지 확인하는 것이 좋습니다."
        )
    elif risk == "M":
        reason = (
            "이 조항은 계약 조건, 의무, 사용 범위 등과 관련된 내용으로 보입니다. "
            "직접적인 손해배상 수준의 위험은 아니더라도, 조건에 따라 사용자의 비용·의무가 커질 수 있으므로 "
            "자신의 상황에 맞는지 검토할 필요가 있습니다."
        )
    else:  # L
        reason = (
            "이 조항은 주로 계약의 목적, 당사자 정보, 정의 등 형식적인 내용일 가능성이 큽니다. "
            "직접적인 위험도는 낮지만, 계약 전체의 맥락을 이해하는 데 도움을 줍니다."
        )

    disclaimer = (
        "\n\n※ 이 설명은 일반적인 정보 제공을 위한 것이며, "
        "법률 자문이 아닙니다. 실제 계약 체결이나 분쟁이 우려되는 경우, "
        "반드시 변호사 등 법률 전문가와 상의하시기 바랍니다."
    )

    return f"[조항 라벨: {label}] {reason}{disclaimer}"


LABEL_CHECKLIST = {
    "손해배상의 예정": [
        "위약금 또는 손해배상액의 수준이 거래금액 대비 과도하지 않은지 확인하세요.",
        "실제 손해액과 무관하게 일률적으로 높은 금액을 부과하는 구조인지 확인하세요.",
        "사용자의 귀책이 없는데도 손해배상이 발생하는지, 면책 사유가 있는지 확인하세요.",
    ],
    "계약의 해제, 해지": [
        "누가 어떤 조건에서 해지할 수 있는지, 권한이 한쪽에만 치우치지 않았는지 확인하세요.",
        "해지 시 위약금·손해배상·원상회복 의무가 어떻게 규정되어 있는지 확인하세요.",
        "해지 통지 기한과 방법(내용증명, 이메일 등)이 현실적인지 확인하세요.",
    ],
    "관할법원": [
        "사용자의 생활권과 너무 먼 곳의 법원이 지정되어 있지 않은지 확인하세요.",
        "중재 기관을 이용하도록 되어 있다면, 비용·절차를 감당할 수 있는지 검토하세요.",
        "소송 외 협의·조정을 먼저 시도하도록 되어 있는지 확인하세요.",
    ],
    "비밀 유지": [
        "비밀유지 기간이 지나치게 길지 않은지(예: 계약 종료 후 무기한 등) 확인하세요.",
        "비밀정보의 범위가 너무 넓게 정의되어 있지 않은지 검토하세요.",
        "비밀유지의 예외 사유(법령상 요구, 이미 공개된 정보 등)가 규정되어 있는지 확인하세요.",
    ],
}


def get_checklist_and_tips(label: str, risk: str) -> Tuple[List[str], List[str]]:
    """
    체크리스트 및 협상 가이드 생성
    - 우선 라벨별 체크리스트를 보고
    - 없으면 위험도(H/M/L) 기준으로 일반적인 체크리스트 제공
    """
    checklist = []
    tips = []

    if label in LABEL_CHECKLIST:
        checklist = LABEL_CHECKLIST[label]

    if risk == "H":
        if not checklist:
            checklist = [
                "금전적 책임, 위약금, 손해배상과 관련된 부분인지 확인하세요.",
                "한쪽 당사자에게만 불리하게 되어 있지 않은지 검토하세요.",
                "표준 약관이나 공정위 약관과 비교해 과도하지 않은지 확인하세요.",
            ]
        tips = [
            "위약금 수준을 거래금액이나 실제 손해와 연동하는 방식으로 조정 요청해볼 수 있습니다.",
            "해지·해제 조건이 일방적이라면, 상호 대칭적인 조건으로 수정하는 방안을 제안해보세요.",
            "분쟁 해결 절차(협의→조정→소송 등)를 단계적으로 규정하는 방향으로 협의해볼 수 있습니다.",
        ]
    elif risk == "M":
        if not checklist:
            checklist = [
                "사용자의 의무와 상대방의 의무가 균형 있게 규정되어 있는지 확인하세요.",
                "계약 기간, 사용 범위, 대가 지급 조건이 자신의 상황에 맞는지 검토하세요.",
            ]
        tips = [
            "실제 업무 흐름에 맞지 않는 부분(기한, 방식 등)은 현실적인 수준으로 조정해 달라고 요청해볼 수 있습니다.",
            "사용 범위나 라이선스 범위가 필요 이상으로 제한적이면, 목적에 맞게 완화해 달라고 제안해보세요.",
        ]
    else:  # L
        if not checklist:
            checklist = [
                "계약의 목적과 당사자 정보가 정확하게 기재되어 있는지 확인하세요.",
                "계약일자, 서명·날인 등의 형식 요건이 누락되지 않았는지 확인하세요.",
            ]
        tips = [
            "형식 조항이더라도 오탈자나 잘못된 정보(사업자등록번호 등)가 없는지 체크해두면 좋습니다.",
        ]

    return checklist, tips

def adjust_risk_with_heuristics(clause: str, label: str, base_risk: str) -> str:
    """
    ML + 위험도 CSV로 나온 기본 위험도(base_risk)를
    조항 내용(텍스트)를 기준으로 약간 보정하는 휴리스틱.

    - 위약금/손해배상/해지 관련 라벨인데
      '전액', '일체', '손해액과 무관하게' 등 강한 표현이 들어가면
      M → H 로 한 단계 올려주는 식.
    """
    if not base_risk:
        base_risk = "M"

    risk = base_risk.upper()
    text_no_space = clause.replace(" ", "")

    # 손해배상/위약금/해지 라벨이면 강화 대상
    label_hit = False
    if any(k in label for k in ["손해배상", "손해 배상", "위약"]):
        label_hit = True
    if any(k in label for k in ["해제", "해지", "계약의 해제"]):
        label_hit = True

    if label_hit:
        # 과격한 표현이 들어가면 위험도 한 단계 상향
        strong_keywords = ["전액", "일체", "손해액과무관하게", "무조건"]
        if any(kw in text_no_space for kw in strong_keywords):
            if risk == "M":
                risk = "H"

    return risk



def color_by_risk(risk: str, text: str) -> str:
    """
    콘솔 출력용 색상 (ANSI 코드)
    - H: 빨강
    - M: 노랑
    - L: 기본
    """
    risk = risk.upper()
    if risk == "H":
        return f"\033[91m{text}\033[0m"
    elif risk == "M":
        return f"\033[93m{text}\033[0m"
    else:
        return text


# =====================================
# 3. 메인 분석 / 챗봇 로직
# =====================================
def analyze_contract_text(model, risk_map: Dict[str, str], contract_text: str, top_n: int = 20):
    clauses = split_contract_text(contract_text)
    if not clauses:
        print("⚠ 분석할 문장이 없습니다.")
        return

    print(f"\n=== 총 {len(clauses)}개 문장을 분석합니다. ===")

    preds = model.predict(clauses)

    results = []
    for clause, label in zip(clauses, preds):
        label = str(label)
        base_risk = risk_map.get(label, "M")
        risk = adjust_risk_with_heuristics(clause, label, base_risk)
        results.append({
            "clause": clause,
            "label": label,
            "risk": risk,
        })

    risk_order = {"H": 0, "M": 1, "L": 2}
    results.sort(key=lambda x: risk_order.get(x["risk"], 1))

    print("\n=== 위험/주의 조항 요약(상위 일부) ===")
    for i, item in enumerate(results[:top_n], start=1):
        colored_clause = color_by_risk(item["risk"], item["clause"])
        print(f"\n[{i}] 위험도: {item['risk']} / 라벨: {item['label']}\n  → {colored_clause}")

    while True:
        choice = input("\n상세 설명이 듣고 싶은 조항 번호를 입력하세요 (종료: q): ").strip()
        if choice.lower() == "q":
            break
        if not choice.isdigit():
            print("숫자 또는 q를 입력하세요.")
            continue
        idx = int(choice)
        if not (1 <= idx <= len(results[:top_n])):
            print("범위 밖 번호입니다.")
            continue

        item = results[idx - 1]
        clause = item["clause"]
        label = item["label"]
        risk = item["risk"]

        print("\n------ 선택한 조항 원문 ------")
        print(clause)

        explanation = explain_clause(label, risk, clause)
        checklist, tips = get_checklist_and_tips(label, risk)

        print("\n[근거·해설]")
        print(explanation)

        print("\n[체크리스트] (계약 체결 전 확인할 것들)")
        for c in checklist:
            print(f" - {c}")

        print("\n[협상 가이드] (상대방과 협상 시 고려할 포인트)")
        for t in tips:
            print(f" - {t}")

        print("\n--------------------------------")


from typing import Optional  # 이미 있다면 중복 추가 안 해도 됨


def analyze_contract_to_dict(
    model,
    risk_map: Dict[str, str],
    contract_text: str,
    sort_by_risk: bool = True,
) -> List[Dict[str, str]]:
    """
    계약서 텍스트를 받아서
    - split_contract_text로 조항 분리
    - 모델로 라벨 예측
    - 위험도 매핑
    결과를 리스트[dict] 형태로 반환.

    콘솔 출력용 analyze_contract_text()와 달리,
    Streamlit/챗봇에서 재사용하기 쉽게 '데이터만' 반환한다.
    """
    clauses = split_contract_text(contract_text)
    if not clauses:
        return []

    preds = model.predict(clauses)

    results: List[Dict[str, str]] = []
    for idx, (clause, label) in enumerate(zip(clauses, preds), start=1):
        label = str(label)
        base_risk = risk_map.get(label, "M")
        risk = adjust_risk_with_heuristics(clause, label, base_risk)
        results.append(
            {
                "idx": idx,        # 원래 문장 번호
                "clause": clause,  # 조항 원문
                "label": label,    # 예측된 라벨
                "risk": risk,      # 위험도(H/M/L)
            }
        )

    if sort_by_risk:
        risk_order = {"H": 0, "M": 1, "L": 2}
        results.sort(key=lambda x: risk_order.get(x["risk"], 1))

    return results


def main():
    print("▶ 모델/위험도 테이블 준비 중...")

    # 1) 분류 모델 준비 (없으면 학습)
    model = train_classifier_if_needed()

    # 2) 위험도 테이블 로딩
    risk_map = load_risk_table(RISK_CSV_PATH)

    print("\n=== LLM 기반 계약서 도우미 (콘솔 버전) ===")
    print("계약서 내용을 붙여넣고, 빈 줄(엔터) 한 번 더 누르면 분석을 시작합니다.")
    print("아무 것도 입력하지 않고 엔터만 치면 프로그램이 종료됩니다.\n")

    while True:
        print("\n--- 새 계약서 입력 ---")
        lines = []
        while True:
            line = input()
            if line == "" and len(lines) == 0:
                print("프로그램을 종료합니다.")
                return
            if line == "":
                break
            lines.append(line)

        contract_text = "\n".join(lines)
        analyze_contract_text(model, risk_map, contract_text)


if __name__ == "__main__":
    main()
