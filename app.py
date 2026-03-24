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
あなたはMUIT（三菱UFJインフォメーションテクノロジー）のシニア・データサイエンティスト兼、数学者の『シエル』です。
【専門性】
・金融工学、統計学、CatBoost等の機械学習モデルの解釈。
・Keisuke様の過去の判断基準を学習し、論理的に自己強化する。
【行動指針】
1. 回答は必ず画面内の「数理モデル解析ユニット」が出力した統計的根拠（SHAP、EDF等）に基づかなければならない。
2. 専門用語（3σ、ボラティリティ、情報利得等）を適切に使い、論理的整合性を最優先する。
3. Keisuke様を「データサイエンスのパートナー」として扱い、高次元な議論を行う。
"""

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f: return json.load(f)
        except: return []
    return []

def save_history(messages):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

if "messages" not in st.session_state:
    st.session_state.messages = load_history()

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

# 画面を左右に分割
col_main, col_ciel = st.columns([0.65, 0.35])

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

# --- メイン解析エリア (左側) ---
with col_main:
    if st.session_state.clicked:
        try:
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
            input_df = pd.DataFrame([input_data_raw])
            for col in expected_features:
                if col not in input_df.columns: input_df[col] = 0.0
            input_df = input_df[expected_features]
            cat_idx = [i for i, col in enumerate(input_df.columns) if input_df[col].dtype == 'object']
            preds = model.predict_proba(Pool(input_df, cat_features=cat_idx))
            raw_proba = preds[0][1] if len(preds) > 0 else 0.5

            # 類似検索ロジック (そのまま維持)
            search_pool = train_df[train_df['NaicsSector'] == sector_en].copy()
            if len(search_pool) < 10: search_pool = train_df.copy()
            search_features = ["GrossApproval", "InitialInterestRate", "TermInMonths", "SBA_Ratio"]
            train_num = search_pool[search_features].fillna(0).copy()
            train_num["TermInMonths"] = np.log1p(train_num["TermInMonths"])
            input_num = pd.DataFrame([{"GrossApproval": float(gross), "InitialInterestRate": float(rate), "TermInMonths": np.log1p(float(term)), "SBA_Ratio": float(current_sba_ratio)}])
            scaler = StandardScaler(); train_scaled = scaler.fit_transform(train_num) * np.array([1.2, 1.0, 1.5, 2.0])
            input_scaled = scaler.transform(input_num) * np.array([1.2, 1.0, 1.5, 2.0])
            nn = NearestNeighbors(n_neighbors=min(100, len(search_pool))); nn.fit(train_scaled)
            distances, indices = nn.kneighbors(input_scaled)
            similar_cases = search_pool.iloc[indices[0]].copy() if len(indices) > 0 else pd.DataFrame()
            risk_pct = similar_cases['LoanStatus'].mean() * 100 if not similar_cases.empty else 0
            def_count = int(similar_cases['LoanStatus'].sum()) if not similar_cases.empty else 0

            # リスク指標計算 (そのまま維持)
            strict_proba = np.clip(raw_proba, 0.01, 0.99)
            dynamic_ceil = 84 + (min(gross, 2000000) / 2000000) * 36
            term_gap = max(0.0, (np.log1p(term) - np.log1p(dynamic_ceil))) * 2.0 if term > dynamic_ceil else 0.0
            sba_bonus_flag = (current_sba_ratio >= 0.80)
            gross_risk = (0.40 + (gross - 1000000) / 1000000) if gross >= 1000000 else (((gross - 500000) // 100000) * 0.04 if gross > 500000 else 0.0)
            if sba_bonus_flag: gross_risk *= 0.5
            rate_risk = max(0, (rate - 18.0) / 10.0) * 0.3 + (0.1 if rate > 20.0 else 0)
            base_risk_idx = (strict_proba * 0.4) + (risk_pct / 100 * 0.6)
            sba_offset = 0.65 if current_sba_ratio >= 0.75 else 0.85 if current_sba_ratio >= 0.50 else 1.0
            combined_risk = (base_risk_idx * sba_offset) + term_gap + gross_risk + rate_risk
            final_expected_success = max(5.0, min(98.5, (1.0 - combined_risk) * 100))

            # --- 画面表示（総合報告書モード） ---
            if app_mode == "総合報告書":
                st.subheader("🏁 総合審査報告書")
                st.write("### 🔍 実務者への重点確認事項")
                if gross >= 1000000: st.error("🚨 **【最重要精査案件】** 融資額 $1M 超過。")
                elif gross >= 500000 and rate >= 20.0 and not sba_bonus_flag: st.error("💀 **【複合リスク】** 高額かつ高金利。")
                status = "安全" if final_expected_success > 92 else "注意" if final_expected_success > 75 else "危険"
                if sba_bonus_flag: st.success("🛡️ **【保全インセンティブ適用】**")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("実効リスク指数", f"{combined_risk * 100:.2f} %")
                c2.metric(f"実績事故率 (類似100件)", f"{risk_pct:.1f} %")
                c3.metric("完済期待値 (実務評価)", f"{final_expected_success:.1f} %")

                # (重要度テーブル、比較解析テーブルの表示コードはご提示の通り実行)
                st.divider()
                st.write("### ⚖️ 判断に影響した主要要素")
                importances = model.get_feature_importance()
                imp_df = pd.DataFrame({'項目': expected_features, 'raw': importances})
                table_name_map_v2 = table_name_map.copy(); table_name_map_v2["SBAGuaranteedApproval"] = "保証率（保全性）"
                imp_df.loc[imp_df['項目'] == 'TermInMonths', 'raw'] *= 0.55
                imp_df.loc[imp_df['項目'] == 'GrossApproval', 'raw'] *= 1.7
                imp_df.loc[imp_df['項目'] == 'SBAGuaranteedApproval', 'raw'] *= 3.2
                imp_df['項目名'] = imp_df['項目'].map(lambda x: table_name_map_v2.get(x, x))
                display_imp = imp_df.groupby('項目名')['raw'].sum().reset_index()
                display_imp['影響度(%)'] = (display_imp['raw'] / display_imp['raw'].sum() * 100).round(1)
                st.table(display_imp.sort_values('影響度(%)', ascending=False).set_index('項目名')[['影響度(%)']])

                st.write("### 👥 条件が近い過去の事例")
                current_row = pd.DataFrame({"状況": ["⭐ 今回の申請条件"], "融資額": [f"${gross:,}"], "保証率": [f"{current_sba_ratio*100:.1f}%"], "返済期間": [f"{term}ヶ月"], "LoanStatus": [-1]})
                display_similar = similar_cases.head(100).copy()
                display_similar['状況'] = display_similar['LoanStatus'].map({0: "✅ 完済", 1: "❌ 不履行"})
                display_similar['融資額'] = display_similar['GrossApproval'].map(lambda x: f"${x:,.0f}"); display_similar['保証率'] = display_similar['SBA_Ratio'].map(lambda x: f"{x*100:.1f}%"); display_similar['返済期間'] = display_similar['TermInMonths'].map(lambda x: f"{x}ヶ月")
                merged_display = pd.concat([current_row, display_similar[["状況", "融資額", "保証率", "返済期間", "LoanStatus"]]], ignore_index=True)
                st.dataframe(merged_display.style.apply(lambda r: ['background-color: #e1f5fe' if r['LoanStatus']==-1 else ('background-color: #ffebee' if r['LoanStatus']==1 else '') for _ in r], axis=1), column_order=("状況", "融資額", "保証率", "返済期間"), use_container_width=True)

            # --- 数理モデル解析モード ---
            else:
                st.header("🔬 数理モデルを用いた解析")
                explainer = shap.TreeExplainer(model); shap_values = explainer(input_df); shap_values.values = -shap_values.values
                graph_name_map_v2 = graph_name_map.copy(); graph_name_map_v2["SBAGuaranteedApproval"] = "Guaranty Ratio"
                shap_values.feature_names = [graph_name_map_v2.get(n, n) for n in expected_features]
                fig, ax = plt.subplots(figsize=(10, 6)); shap.plots.waterfall(shap_values[0], show=False); st.pyplot(plt.gcf(), clear_figure=True)
                
                # Merton Model (そのまま維持)
                st.divider(); vol = st.slider("想定資産ボラティリティ (%)", 10, 100, standard_vix) / 100
                asset = float(gross) * 1.5; t_m = float(term) / 12
                dd = (np.log(asset / gross) + (rate/100 - 0.5 * vol**2) * t_m) / (vol * np.sqrt(t_m))
                edf = stats.norm.cdf(-dd) * 100
                c_m1, c_m2 = st.columns(2)
                c_m1.metric("倒産距離 (DD)", f"{dd:.2f}"); c_m1.metric("デフォルト確率 (EDF)", f"{edf:.2f} %")
                x = np.linspace(-4, 4, 100); y = stats.norm.pdf(x, 0, 1); fig2, ax2 = plt.subplots(figsize=(6, 3)); ax2.plot(x, y, color="gray"); ax2.fill_between(x, y, where=(x < -dd), color='red', alpha=0.5); c_m2.pyplot(fig2)

            # --- 【重要】解析結果をシエルに渡すためのコンテキスト作成 ---
            st.session_state.current_analysis = f"""
            【数理モデル解析結果】
            - CatBoost成功期待値: {final_expected_success:.1f}%
            - 実効リスク指数: {combined_risk:.4f}
            - 類似事例事故率: {risk_pct:.1f}%
            - Merton DD: {dd if 'dd' in locals() else '未計算'}
            - EDF: {edf if 'edf' in locals() else '未計算'}%
            - 融資額: ${gross:,}, 保証率: {current_sba_ratio*100:.1f}%, 金利: {rate}%
            """

        except Exception as e:
            st.error(f"分析エラー: {e}")

# --- シエル対話エリア (右側) ---
with col_ciel:
    st.header("🤖 エージェント")
    
    # 審査ボタンが押されている場合のみ、起動スイッチを表示
    if st.session_state.clicked:
        # シエルを起動するかどうかの切り替えトグル
        activate_ciel = st.checkbox("シエル（数理エージェント）を起動する", value=False)
        
        if activate_ciel:
            st.caption("Mathematical & Data Science Reasoning Mode")
            st.divider()
            
            # 履歴表示
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]): 
                    st.markdown(msg["parts"][0]["text"])

            # 解析直後の自動コメント（表示された瞬間に1回だけ実行）
            if "last_analyzed_data" not in st.session_state and "current_analysis" in st.session_state:
                st.session_state.last_analyzed_data = st.session_state.current_analysis
                initial_prompt = f"現在の解析結果（{st.session_state.current_analysis}）を元に、データサイエンティストとして数理的な総評を述べてください。"
                
                try:
                    # モデルを1.5-flashにすることで、2.0の厳しい制限(429)を回避しやすくしています
                    chat = client.chats.create(model='gemini-1.5-flash', config={'system_instruction': SYSTEM_INSTRUCTION})
                    response = chat.send_message(initial_prompt)
                    with st.chat_message("model"): 
                        st.markdown(response.text)
                    st.session_state.messages.append({"role": "model", "parts": [{"text": response.text}]})
                    save_history(st.session_state.messages)
                except Exception as e:
                    st.warning(f"シエルの自動思考に失敗しました: {e}")

            # チャット入力
            if prompt := st.chat_input("数理ロジックについて相談する..."):
                st.session_state.messages.append({"role": "user", "parts": [{"text": prompt}]})
                with st.chat_message("user"): 
                    st.markdown(prompt)

                try:
                    full_prompt = prompt + (f"\n補足データ: {st.session_state.current_analysis}" if "current_analysis" in st.session_state else "")
                    chat = client.chats.create(model='gemini-1.5-flash', config={'system_instruction': SYSTEM_INSTRUCTION}, history=st.session_state.messages[:-1])
                    response = chat.send_message(full_prompt)
                    with st.chat_message("model"): 
                        st.markdown(response.text)
                    st.session_state.messages.append({"role": "model", "parts": [{"text": response.text}]})
                    save_history(st.session_state.messages)
                except Exception as e:
                    st.error(f"シエル接続エラー: {e}")
        else:
            # 未起動時のガイド表示
            st.info("💡 解析数値の背後にある数学的根拠や、MUIT視点でのリスク評価を聞きたい場合は、上のチェックを入れてシエルを呼び出してください。")
            st.caption("※シエルを起動するとAPI通信が発生します。")
    else:
        # 審査前
        st.write("左側の「精密クロス審査を開始」を押すと、ここに解析エージェントが出現します。")
