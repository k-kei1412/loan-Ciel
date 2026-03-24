# --- 前半のインポートと基本設定はそのまま ---
import streamlit as st
import pandas as pd
import numpy as np
import os
import json # 履歴保存用に追加
import matplotlib.pyplot as plt
import shap
import scipy.stats as stats
from catboost import CatBoostClassifier, Pool
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.font_manager as fm
from google import genai # シエル（Gemini API）

# 1. ページ設定
st.set_page_config(page_title="ローン審査AI：真・完全体 ✕ シエル", layout="wide")

# --- 【追加】シエル（AI）の基本設定 ---
ST_KEY = "GOOGLE_API_KEY"
HISTORY_FILE = "persistent_history.json"
SYSTEM_INSTRUCTION = """
あなたはメガバンクのシニア・データサイエンティスト兼、数学者の『シエル』です。
【専門性】
・金融工学、統計学、確率論、CatBoost等の機械学習モデルの解釈。
・Keiの過去の判断基準を学習し、論理的に自己強化する。
・2次元から3次元までの数学的データ的にとらえられる
【行動指針】
1. 回答は必ず画面内の「数理モデル解析ユニット」が出力した統計的根拠（SHAP、EDF等）に基づかなければならない。
2. 専門用語（3σ、ボラティリティ、情報利得等）を適切に使い、論理的整合性を最優先する。
3. Keisuke様を「データサイエンスのパートナー」として扱い、高次元な議論を行う。
"""
# --- 履歴保存をメモリのみに変更 ---
def load_history():
    return [] # 常に空で開始（ブラウザを閉じれば消える設定）

def save_history(messages):
    pass # ファイルに保存しない

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f: return json.load(f)
        except: return []
    return []

def save_history(messages):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

# 冒頭の初期化部分をこのように修正
if "messages" not in st.session_state:
    st.session_state.messages = load_history()

# 追加：もしmessagesがNoneだった場合の保険
if st.session_state.messages is None:
    st.session_state.messages = []

# APIクライアント初期化
api_key = st.secrets.get(ST_KEY)
client = genai.Client(api_key=api_key)

# --- (以下、フォント設定・業界マップ・リソース読み込み・名前マップは全てそのまま維持) ---
def set_japanese_font():
    fonts = ['Heiti TC', 'MS Gothic', 'Hiragino Sans', 'Yu Gothic', 'IPAexGothic', 'Noto Sans CJK JP', 'DejaVu Sans']
    for f in fonts:
        if f in [font.name for font in fm.fontManager.ttflist]:
            plt.rcParams['font.family'] = f
            plt.rcParams['axes.unicode_minus'] = False
            break
set_japanese_font()

sectors_map = {
    "accommodation": "宿泊・飲食サービス業", "administrative": "運営支援・廃棄物処理",
    "agriculture": "農業・林業・漁業", "arts": "芸術・娯楽・レクリエーション",
    "construction": "建設業", "educational": "教育サービス業",
    "finance": "金融業・保険業", "health": "医療・福祉",
    "information": "情報通信業", "management": "企業管理・持株会社",
    "manufacturing": "製造業", "mining": "採鉱・石油ガス採掘",
    "professional": "専門・科学・技術サービス", "public": "公務",
    "real estate": "不動産・賃貸業", "retail": "小売業",
    "transportation": "運輸業・倉庫業", "utilities": "公益事業", "wholesale": "卸売業"
}

sector_vix_map = {
    "accommodation": 45, "administrative": 40, "agriculture": 23, 
    "arts": 50, "construction": 30, "educational": 25,
    "finance": 23, "health": 20, "information": 55, 
    "management": 60, "manufacturing": 30, "mining": 30,
    "professional": 45, "public": 18, "real estate": 40, 
    "retail": 40, "transportation": 30, "utilities": 18, "wholesale": 30
}

@st.cache_resource
def load_resources():
    model = CatBoostClassifier()
    model.load_model("catboost_model.cbm")
    try:
        target = "train.csv" if os.path.exists("train.csv") else "train (4).csv"
        df = pd.read_csv(target)
        df['NaicsSector'] = df['NaicsSector'].astype(str)
        df['SBA_Ratio'] = (df['SBAGuaranteedApproval'] / df['GrossApproval']).fillna(0)
        return model, df, target
    except:
        return model, pd.DataFrame(), "None"

model, train_df, file_name = load_resources()
expected_features = model.feature_names_

graph_name_map = {
    "TermInMonths": "Loan Term", "GrossApproval": "Loan Amount", "InitialInterestRate": "Interest Rate", 
    "NaicsSector": "Industry Sector", "SBAGuaranteedApproval": "SBA Guaranty", "CollateralInd": "Collateral",
    "ApprovalFiscalYear": "Fiscal Year", "Subprogram": "Subprogram", "FixedOrVariableInterestInd": "Rate Type",
    "BusinessAge": "Business Age", "CongressionalDistrict": "Location Code", "BusinessType": "Business Type", "JobsSupported": "Jobs Created",
    "SBA_Ratio": "Guaranty Rate"
}

table_name_map = {
    "TermInMonths": "返済期間", "GrossApproval": "融資額", "InitialInterestRate": "金利", 
    "NaicsSector": "業界セクター", "SBAGuaranteedApproval": "保証額", "CollateralInd": "担保有無",
    "BusinessAge": "事業歴", "BusinessType": "法人形態", "JobsSupported": "雇用創出数",
    "Subprogram": "支援プログラム", "FixedOrVariableInterestInd": "金利タイプ", "CongressionalDistrict": "地域区分（所在地区）",
    "SBA_Ratio": "保証率", "ApprovalFiscalYear": "承認年度", "RevolverStatus": "反復利用の有無"
}

def get_japanese_sector(en_text):
    text = str(en_text).lower()
    if "other" in text: return "その他サービス業"
    for k, v in sectors_map.items():
        if k in text: return v
    return en_text

if "clicked" not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True

# --- UI構成 ---
st.title("🏦 中小企業向けローン返済予測 AIシステム")

# 以前の st.columns([0.65, 0.35]) を削除し、1カラム構成にします。
# これにより、シエルは自動的に解析結果の下に配置されます。

with st.sidebar:
    st.header("📋 申請者情報入力")
    app_mode = st.sidebar.radio("📊 表示モード切替", ["総合報告書", "数理モデル解析"])
    st.divider()
    gross = st.number_input("融資額 ($)", 0, 10000000, 500000)
    sba = st.number_input("保証額 ($)", 0, 10000000, 300000)
    rate = st.number_input("金利 (%)", 0.0, 35.0, 15.0)
    term = st.number_input("返済期間 (月)", 1, 360, 84)
    b_age = st.selectbox("事業歴", ["2年以上 (Existing)", "2年未満 (New Business)"])
    b_age_val = "Existing or more than 2 years old" if "2年以上" in b_age else "New Business or less than 2 years old"
    b_type = st.selectbox("法人形態", ["株式会社 (CORPORATION)", "個人事業主 (INDIVIDUAL)", "パートナーシップ (PARTNERSHIP)"])
    b_type_val = b_type.split("(")[1].replace(")", "")
    
    if not train_df.empty:
        unique_en_sectors = sorted(train_df['NaicsSector'].unique())
        display_options = [get_japanese_sector(s) for s in unique_en_sectors]
        selected_jp = st.selectbox("産業セクター", options=display_options)
        sector_en = unique_en_sectors[display_options.index(selected_jp)]
        vix_key = next((k for k, v in sectors_map.items() if v == selected_jp), "")
        standard_vix = sector_vix_map.get(vix_key, 30)
        st.info(f"💡 この業界の標準ボラティリティは **{standard_vix}%** です")
    else:
        sector_en = "Finance_insurance"
        st.selectbox("産業セクター", options=["データ未読み込み"])

    jobs = st.slider("現在の雇用員数", 0, 500, 5)
    rate_type = st.radio("金利タイプ", ["変動金利 (V)", "固定金利 (F)"])
    rate_type_val = "V" if "変動" in rate_type else "F"
    collateral = st.selectbox("担保の有無", ["あり (Y)", "なし (N)"])
    collateral_val = "Y" if "あり" in collateral else "N"
    
    submit = st.button("精密クロス審査を開始", on_click=click_button)

# --- メイン解析エリア ---
# col_main の中身を通常の st 空間に配置（全幅表示）
if st.session_state.clicked:
    try:
        # (解析ロジックは変更なしのため省略。中身はそのまま維持してください)
        current_sba_ratio = sba / gross if gross > 0 else 0
        input_data_raw = {
            "GrossApproval": float(gross), "SBAGuaranteedApproval": float(sba),
            "InitialInterestRate": float(rate), "TermInMonths": float(term),
            "NaicsSector": str(sector_en), "ApprovalFiscalYear": 2024.0, 
            "Subprogram": "Guaranty", "FixedOrVariableInterestInd": rate_type_val, 
            "CongressionalDistrict": 10.0, "BusinessType": b_type_val, 
            "BusinessAge": b_age_val, "RevolverStatus": 0.0, 
            "JobsSupported": float(jobs), "CollateralInd": str(collateral_val)
        }
        input_df = pd.DataFrame([input_data_raw])[expected_features]
        cat_idx = [i for i, col in enumerate(input_df.columns) if input_df[col].dtype == 'object']
        preds = model.predict_proba(Pool(input_df, cat_features=cat_idx))
        raw_proba = preds[0][1] if len(preds) > 0 else 0.5

        # 類似検索
        search_pool = train_df[train_df['NaicsSector'] == sector_en].copy()
        if len(search_pool) < 10: search_pool = train_df.copy()
        search_features = ["GrossApproval", "InitialInterestRate", "TermInMonths", "SBA_Ratio"]
        train_num = search_pool[search_features].fillna(0).copy()
        train_num["TermInMonths"] = np.log1p(train_num["TermInMonths"])
        input_num = pd.DataFrame([{"GrossApproval": float(gross), "InitialInterestRate": float(rate), "TermInMonths": np.log1p(float(term)), "SBA_Ratio": float(current_sba_ratio)}])
        scaler = StandardScaler(); train_scaled = scaler.fit_transform(train_num); input_scaled = scaler.transform(input_num)
        nn = NearestNeighbors(n_neighbors=min(100, len(search_pool))); nn.fit(train_scaled)
        distances, indices = nn.kneighbors(input_scaled)
        similar_cases = search_pool.iloc[indices[0]].copy()
        risk_pct = similar_cases['LoanStatus'].mean() * 100

        # リスク計算
        strict_proba = np.clip(raw_proba, 0.01, 0.99)
        combined_risk = (strict_proba * 0.4) + (risk_pct / 100 * 0.6) # 簡易化
        final_expected_success = max(5.0, min(98.5, (1.0 - combined_risk) * 100))
        # Merton DD
        vol = 30 / 100; asset = float(gross) * 1.5; t_m = float(term) / 12
        dd = (np.log(asset / gross) + (rate/100 - 0.5 * vol**2) * t_m) / (vol * np.sqrt(t_m))
        edf = stats.norm.cdf(-dd) * 100

        if app_mode == "総合報告書":
            st.subheader("🏁 総合審査報告書")
            c1, c2, c3 = st.columns(3)
            c1.metric("実効リスク指数", f"{combined_risk * 100:.2f} %")
            c2.metric("実績事故率", f"{risk_pct:.1f} %")
            c3.metric("完済期待値", f"{final_expected_success:.1f} %")
            st.write("### 👥 近しい過去事例（上位100件）")
            st.dataframe(similar_cases.head(100)[["LoanStatus", "GrossApproval", "SBA_Ratio", "TermInMonths"]], use_container_width=True)
        else:
            st.header("🔬 数理モデル解析")
            st.metric("倒産距離 (DD)", f"{dd:.2f}")
            # SHAP等の表示（省略せず維持してください）

        # コンテキスト保存
        st.session_state.current_analysis = f"成功期待値:{final_expected_success:.1f}%, DD:{dd:.2f}, EDF:{edf:.2f}%"

    except Exception as e:
        st.error(f"分析エラー: {e}")

st.divider()

# --- シエル対話エリア (下部配置) ---
st.header("AI分析ボット：シエル")

if st.session_state.clicked:
    # 履歴削除ボタン
    btn_col1, btn_col2 = st.columns([0.8, 0.2])
    with btn_col2:
        if st.button("💬 履歴をリセット"):
            st.session_state.messages = []
            if "last_analyzed_data" in st.session_state:
                del st.session_state.last_analyzed_data
            # 物理ファイルを使わない設定にしている場合は、ここだけでOK
            st.rerun()

    activate_ciel = st.checkbox("シエルを起動して対話を開始する", value=False)
    
    if activate_ciel:
        st.info("Mathematical Reasoning Mode: ON")
        
        # --- ここからが「連動型」のロジック ---
        
        # 1. データ更新チェック
        is_data_updated = (
            "last_analyzed_data" not in st.session_state or 
            st.session_state.last_analyzed_data != st.session_state.current_analysis
        )

        # 2. 過去の履歴を表示
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): 
                st.markdown(msg["parts"][0]["text"])

        # 3. データが新しい場合のみ、シエルに再考させる
        if is_data_updated and "current_analysis" in st.session_state:
            st.session_state.last_analyzed_data = st.session_state.current_analysis
            initial_prompt = f"入力パラメータが更新されました。最新の解析結果（{st.session_state.current_analysis}）に基づき、数理的な再評価を述べてください。"
            
            try:
                # model名を 'gemini-1.5-flash' に統一
                chat = client.chats.create(model='models/gemini-flash-latest', config={'system_instruction': SYSTEM_INSTRUCTION})
                response = chat.send_message(initial_prompt)
                
                with st.chat_message("model"): 
                    st.markdown(response.text)
                
                # 履歴に追加
                st.session_state.messages.append({"role": "model", "parts": [{"text": response.text}]})
            except Exception as e:
                st.warning(f"シエルの再考に失敗しました: {e}")

        # 4. ユーザーからの自由入力
        if prompt := st.chat_input("この解析結果について質問する..."):
            st.session_state.messages.append({"role": "user", "parts": [{"text": prompt}]})
            with st.chat_message("user"): st.markdown(prompt)
            try:
                chat = client.chats.create(model='models/gemini-flash-latest', config={'system_instruction': SYSTEM_INSTRUCTION}, history=st.session_state.messages[:-1])
                response = chat.send_message(prompt + f"\nContext: {st.session_state.current_analysis}")
                with st.chat_message("model"): st.markdown(response.text)
                st.session_state.messages.append({"role": "model", "parts": [{"text": response.text}]})
            except Exception as e:
                st.error(f"シエル接続エラー: {e}")
    else:
        st.write("解析結果についての議論が必要な場合は、上のスイッチをONにしてください。")
else:
    st.write("審査を開始すると、ここにシエルが出現します。")
