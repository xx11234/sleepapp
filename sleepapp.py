import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="SleepWell AI | 智能睡眠健康分析平台",
    layout="wide",
    page_icon="🌙",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.sleepwell.ai/help',
        'Report a bug': "https://www.sleepwell.ai/feedback",
        'About': "# SleepWell AI · 专业睡眠健康分析工具\n## 基于多模型机器学习的风险评估系统"
    }
)

# ==================== 全局样式（完全保留原有） ====================
hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stAppHeader {display: none;}
        .stDeployButton {display: none;}
        ::-webkit-scrollbar {width: 6px; height: 6px;}
        ::-webkit-scrollbar-track {background: #eef2f7; border-radius: 10px;}
        ::-webkit-scrollbar-thumb {background: #b9d0e5; border-radius: 10px;}
        ::-webkit-scrollbar-thumb:hover {background: #7fa0bc;}
        html, body, [class*="css"] {font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;}
        .stTabs [data-baseweb="tab-list"] {gap: 8px; background: transparent; border-bottom: none;}
        .stTabs [data-baseweb="tab"] {border-radius: 40px; padding: 8px 20px; background: rgba(255,255,255,0.6); backdrop-filter: blur(4px); border: none; font-weight: 500; transition: all 0.2s;}
        .stTabs [aria-selected="true"] {background: white; box-shadow: 0 4px 12px rgba(0,0,0,0.05); color: #1a6bb5; border-bottom: 2px solid #1a6bb5;}
        .stButton button {border-radius: 40px; transition: transform 0.2s, box-shadow 0.2s; font-weight: 500;}
        .stButton button:hover {transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0,0,0,0.1);}
        .stApp {background: radial-gradient(circle at 20% 30%, #f4f9fe, #eef3fc);}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ==================== 进入页面（独立全屏Landing Page） ====================
if "entered" not in st.session_state:
    st.session_state.entered = False

if not st.session_state.entered:
    # 隐藏侧边栏和页面边距
    st.markdown("""
    <style>
        section[data-testid="stSidebar"] {display: none !important;}
        .main .block-container {padding-top: 0 !important; max-width: 100% !important;}
        .stApp > .main {display: flex; align-items: center; justify-content: center;}
        .stApp {background: linear-gradient(135deg, #e8f4ff 0%, #d0eaff 100%);}
    </style>
    """, unsafe_allow_html=True)

    # 白色+蓝色配色的进入页 HTML/CSS
    landing_html = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Inter', sans-serif; }
        .landing-container {
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            background: linear-gradient(135deg, #eef6ff 0%, #e0edff 100%);
            overflow: hidden;
        }
        /* 动态光晕背景（浅蓝/白色） */
        .glow-1 {
            position: absolute;
            top: -20%;
            left: -10%;
            width: 70%;
            height: 80%;
            background: radial-gradient(circle, rgba(0,120,255,0.1), transparent);
            border-radius: 50%;
            filter: blur(80px);
            animation: float1 12s infinite alternate;
        }
        .glow-2 {
            position: absolute;
            bottom: -15%;
            right: -5%;
            width: 60%;
            height: 70%;
            background: radial-gradient(circle, rgba(0,80,200,0.08), transparent);
            border-radius: 50%;
            filter: blur(90px);
            animation: float2 14s infinite alternate-reverse;
        }
        .glow-3 {
            position: absolute;
            top: 30%;
            left: 40%;
            width: 50%;
            height: 50%;
            background: radial-gradient(circle, rgba(0,160,255,0.06), transparent);
            border-radius: 50%;
            filter: blur(100px);
            animation: pulse 8s infinite alternate;
        }
        @keyframes float1 {
            0% { transform: translate(0, 0) scale(1); opacity: 0.4; }
            100% { transform: translate(5%, 5%) scale(1.1); opacity: 0.7; }
        }
        @keyframes float2 {
            0% { transform: translate(0, 0) scale(1); opacity: 0.3; }
            100% { transform: translate(-5%, -5%) scale(1.2); opacity: 0.6; }
        }
        @keyframes pulse {
            0% { opacity: 0.1; transform: scale(1); }
            100% { opacity: 0.3; transform: scale(1.3); }
        }
        /* 主卡片（白色玻璃态） */
        .hero-card {
            background: rgba(255, 255, 255, 0.92);
            backdrop-filter: blur(16px);
            border-radius: 56px;
            padding: 3rem 3rem;
            max-width: 1000px;
            width: 85%;
            box-shadow: 0 30px 50px -20px rgba(0,0,0,0.15), 0 0 0 1px rgba(255,255,255,0.6);
            text-align: center;
            z-index: 10;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .hero-card:hover {
            transform: translateY(-6px);
            box-shadow: 0 40px 60px -20px rgba(0,80,150,0.2), 0 0 0 1px rgba(255,255,255,0.8);
        }
        .icon-glow {
            font-size: 5.5rem;
            filter: drop-shadow(0 4px 12px rgba(0,120,255,0.2));
            animation: iconPulse 2s infinite alternate;
        }
        @keyframes iconPulse {
            0% { filter: drop-shadow(0 4px 8px rgba(0,120,255,0.1)); transform: scale(1); }
            100% { filter: drop-shadow(0 8px 20px rgba(0,120,255,0.3)); transform: scale(1.05); }
        }
        .title-gradient {
            font-size: 3.8rem;
            font-weight: 800;
            background: linear-gradient(125deg, #0072e3, #0052a3);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
            letter-spacing: -0.02em;
        }
        .subtitle {
            font-size: 1.3rem;
            color: #2c5a7a;
            margin-bottom: 2rem;
            font-weight: 500;
        }
        .badge-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            justify-content: center;
            margin-bottom: 2rem;
        }
        .badge {
            background: rgba(0,100,200,0.08);
            border-radius: 60px;
            padding: 0.6rem 1.6rem;
            font-weight: 500;
            font-size: 1rem;
            color: #0066cc;
            border: 1px solid rgba(0,100,200,0.15);
            transition: all 0.2s;
        }
        .badge:hover {
            background: rgba(0,100,200,0.15);
            transform: translateY(-2px);
            color: #004999;
            border-color: rgba(0,100,200,0.3);
        }
        .model-card {
            background: rgba(0,80,150,0.05);
            border-radius: 32px;
            padding: 1.2rem;
            margin-bottom: 2rem;
            border: 1px solid rgba(0,100,200,0.1);
        }
        .model-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1.2rem;
            text-align: center;
        }
        .model-item {
            background: rgba(255,255,255,0.7);
            border-radius: 28px;
            padding: 1rem;
            transition: all 0.2s;
            border: 1px solid rgba(0,100,200,0.08);
        }
        .model-item:hover {
            background: white;
            transform: translateY(-4px);
            box-shadow: 0 8px 20px rgba(0,100,200,0.1);
            border-color: rgba(0,100,200,0.2);
        }
        .model-name {
            font-weight: 700;
            font-size: 1.2rem;
            color: #0052a3;
        }
        .model-desc {
            font-size: 0.85rem;
            color: #4a6a8a;
            margin-top: 0.3rem;
        }
        .disclaimer {
            margin-top: 2rem;
            font-size: 0.7rem;
            color: #6c86a0;
        }
        /* 响应式 */
        @media (max-width: 768px) {
            .title-gradient { font-size: 2.5rem; }
            .subtitle { font-size: 1rem; }
            .hero-card { padding: 2rem 1.5rem; }
        }
    </style>
    <div class="landing-container">
        <div class="glow-1"></div>
        <div class="glow-2"></div>
        <div class="glow-3"></div>
        <div class="hero-card">
            <div class="icon-glow">🌙✨</div>
            <h1 class="title-gradient">SleepWell AI</h1>
            <p class="subtitle">智能睡眠健康分析 · 基于AI的风险评估与个性化建议</p>
            <div class="badge-grid">
                <div class="badge">🎯 多模型自由选择</div>
                <div class="badge">📈 实时模拟反馈</div>
                <div class="badge">💡 精准医学级建议</div>
                <div class="badge">📊 历史追踪分析</div>
            </div>
            <div class="model-card">
                <div class="model-grid">
                    <div class="model-item"><div class="model-name">随机森林</div><div class="model-desc">高准确率 · 鲁棒性强</div></div>
                    <div class="model-item"><div class="model-name">逻辑回归</div><div class="model-desc">可解释性强 · 快速</div></div>
                    <div class="model-item"><div class="model-name">线性SVM</div><div class="model-desc">边界清晰 · 适合高维</div></div>
                </div>
            </div>
            <p class="disclaimer">⚠️ 医疗免责声明：本工具仅供健康参考，不构成医疗诊断。</p>
        </div>
    </div>
    """
    st.markdown(landing_html, unsafe_allow_html=True)

    # 按钮样式：蓝色渐变，与白色卡片搭配
    st.markdown("""
    <style>
        div.stButton > button {
            background: linear-gradient(125deg, #1a6bb5, #0e4b78);
            color: white;
            border: none;
            padding: 0.8rem 2rem;
            border-radius: 60px;
            font-weight: 600;
            font-size: 1.1rem;
            width: 100%;
            transition: all 0.3s;
            box-shadow: 0 6px 14px rgba(0,80,150,0.2);
        }
        div.stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 25px rgba(0,100,200,0.3);
            background: linear-gradient(125deg, #2a7bc5, #1a5a88);
        }
        .stColumns {
            margin-top: -1rem;
            margin-bottom: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # 按钮布局
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✨ 探索您的睡眠健康 ✨", use_container_width=True, key="enter_btn"):
            st.session_state.entered = True
            st.rerun()

    st.stop()

# ==================== 进入主应用后恢复侧边栏 ====================
st.markdown("""
<style>
    section[data-testid="stSidebar"] {display: block !important;}
    .main .block-container {padding-top: 1rem !important; max-width: 1200px !important; margin: 0 auto;}
</style>
""", unsafe_allow_html=True)

# ==================== 页面状态（新增） ====================
if "current_page" not in st.session_state:
    st.session_state.current_page = "仪表板"

# ==================== 自定义导航栏（保留样式，改用可点击按钮） ====================
# 自定义CSS：让按钮与原来的链接外观完全一致
st.markdown("""
<style>
    /* 覆盖默认按钮样式，模拟链接 */
    .nav-link-button > button {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
        color: #7e95ae !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        box-shadow: none !important;
        transition: color 0.2s;
        line-height: normal;
    }
    .nav-link-button > button:hover {
        color: #1f6392 !important;
        transform: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    /* 右侧“个人档案”样式 */
    .profile-button > button {
        background: white !important;
        padding: 0.3rem 1rem !important;
        border-radius: 40px !important;
        font-size: 0.8rem !important;
        border: 1px solid #cddfeb !important;
        color: inherit !important;
        font-weight: normal !important;
    }
    .profile-button > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# 使用列布局模拟原导航栏布局
col_logo, col_nav, col_profile = st.columns([2, 3, 1])
with col_logo:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px;">
        <span style="font-size: 2rem;">🌙</span>
        <span style="font-weight: 700; font-size: 1.6rem; background: linear-gradient(125deg, #1a6bb5, #0e4b78); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;">SleepWell AI</span>
        <span style="background: #eef2fa; padding: 0.2rem 0.8rem; border-radius: 40px; font-size: 0.7rem; font-weight: 500; color: #2c6280;">Beta 2.0</span>
    </div>
    """, unsafe_allow_html=True)

with col_nav:
    # 水平排列4个按钮
    nav_items = ["仪表板", "分析报告", "健康指南", "关于我们"]
    btn_cols = st.columns(len(nav_items))
    for idx, page in enumerate(nav_items):
        with btn_cols[idx]:
            st.markdown('<div class="nav-link-button">', unsafe_allow_html=True)
            if st.button(page, key=f"nav_{page}", use_container_width=True):
                st.session_state.current_page = page
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

with col_profile:
    st.markdown('<div class="profile-button">', unsafe_allow_html=True)
    if st.button("👤 个人档案", key="nav_profile", use_container_width=True):
        st.session_state.current_page = "个人档案"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# 分隔线（与原设计一致）
st.markdown("<hr style='margin: 0.5rem 0 1rem 0; opacity: 0.3;'>", unsafe_allow_html=True)

# ==================== 数据加载与预处理（完全不变） ====================
@st.cache_data
def load_data():
    df = pd.read_csv('sleep_health_dataset.csv').dropna()
    df['risk_label'] = (df['sleep_disorder_risk'] != 'Healthy').astype(int)
    return df

@st.cache_data
def preprocess_data(df):
    categorical_cols = ['gender', 'occupation', 'country', 'chronotype',
                        'mental_health_condition', 'season', 'day_type']
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    candidate_num_cols = ['age', 'bmi', 'sleep_duration_hrs', 'sleep_quality_score', 'rem_percentage',
                          'deep_sleep_percentage', 'sleep_latency_mins', 'wake_episodes_per_night',
                          'caffeine_mg_before_bed', 'alcohol_units_before_bed', 'screen_time_before_bed_mins',
                          'steps_that_day', 'nap_duration_mins', 'stress_score', 'work_hours_that_day',
                          'heart_rate_resting_bpm', 'room_temperature_celsius', 'weekend_sleep_diff_hrs',
                          'cognitive_performance_score', 'exercise_day', 'aid', 'shift']
    num_cols = [c for c in candidate_num_cols if c in df.columns]
    feature_cols = num_cols + [c + '_encoded' for c in categorical_cols]
    return df, encoders, num_cols, feature_cols

df_raw = load_data()
df_processed, encoders, num_cols, feature_cols = preprocess_data(df_raw.copy())

X = df_processed[feature_cols]
y = df_processed['risk_label']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

healthy_df = df_processed[df_processed['risk_label'] == 0]
default_vals = {col: healthy_df[col].median() for col in num_cols}

# ==================== 聚类模型 ====================
@st.cache_resource
def train_clustering():
    healthy_X = healthy_df[feature_cols]
    healthy_scaled = scaler.transform(healthy_X)
    pca = PCA(n_components=0.95, random_state=42)
    healthy_pca = pca.fit_transform(healthy_scaled)
    sil_scores = []
    for k in range(2, 6):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(healthy_pca)
        if len(set(labels)) > 1:
            from sklearn.metrics import silhouette_score
            sil = silhouette_score(healthy_pca, labels)
            sil_scores.append(sil)
        else:
            sil_scores.append(-1)
    best_k = range(2, 6)[np.argmax(sil_scores)]
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(healthy_pca)
    cluster_centers = kmeans.cluster_centers_
    cluster_centers_orig = pca.inverse_transform(cluster_centers)
    cluster_profiles = []
    for i in range(best_k):
        center = cluster_centers_orig[i]
        sleep_quality = center[feature_cols.index('sleep_quality_score')] if 'sleep_quality_score' in feature_cols else 0
        stress = center[feature_cols.index('stress_score')] if 'stress_score' in feature_cols else 0
        profile = f"类型{i+1}: " + ("优质睡眠型" if sleep_quality > 7 else "普通睡眠型") + \
                  (" · 压力较低" if stress < 4 else " · 压力中等")
        cluster_profiles.append(profile)
    return kmeans, pca, best_k, cluster_profiles

kmeans_model, pca_model, n_clusters, cluster_profiles = train_clustering()

def predict_user_cluster(user_scaled):
    user_pca = pca_model.transform(user_scaled.reshape(1, -1))
    cluster = kmeans_model.predict(user_pca)[0]
    return cluster, cluster_profiles[cluster]

# ==================== 多模型训练 ====================
@st.cache_resource
def train_model(model_choice, use_smote):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    if model_choice == "随机森林":
        model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
    elif model_choice == "逻辑回归":
        model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    elif model_choice == "线性SVM":
        model = LinearSVC(class_weight='balanced', max_iter=2000, dual='auto', random_state=42)
    else:
        model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X_test)
        y_proba = 1 / (1 + np.exp(-decision))
    else:
        y_proba = None
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    cm = confusion_matrix(y_test, y_pred)
    return model, acc, auc, cm, X_test, y_test, y_proba

# ==================== 侧边栏（完全不变） ====================
with st.sidebar:
    st.markdown("## 🌙 健康档案配置")
    st.markdown("---")
    st.markdown("### 🧠 AI 分析引擎")
    model_choice = st.selectbox("选择预测模型", ["随机森林", "逻辑回归", "线性SVM"], index=0,
                                help="不同模型对数据敏感度不同，随机森林通常表现最佳")
    use_smote = st.checkbox("使用SMOTE平衡样本 (提升少数类识别)", value=False,
                            help="当数据类别不均衡时，显著提高风险人群召回率")
    st.markdown("---")

    with st.expander("👤 基本信息", expanded=True):
        age = st.slider("年龄", 18, 80, int(default_vals.get('age', 30)))
        gender = st.selectbox("性别", encoders['gender'].classes_)
        occupation = st.selectbox("职业", encoders['occupation'].classes_)
        country = st.selectbox("国家/地区", encoders['country'].classes_)
    with st.expander("😴 睡眠核心指标", expanded=True):
        sleep_duration = st.slider("睡眠时长 (小时)", 3.0, 10.0, float(default_vals.get('sleep_duration_hrs', 7.0)), 0.1)
        sleep_quality = st.slider("睡眠质量 (1-10)", 1, 10, int(default_vals.get('sleep_quality_score', 7)))
        rem = st.slider("REM睡眠百分比", 10, 40, int(default_vals.get('rem_percentage', 20)))
        deep = st.slider("深睡百分比", 5, 35, int(default_vals.get('deep_sleep_percentage', 15)))
        latency = st.slider("入睡潜伏期 (分钟)", 1, 60, int(default_vals.get('sleep_latency_mins', 15)))
        wake = st.slider("夜间醒来次数", 0, 10, int(default_vals.get('wake_episodes_per_night', 2)))
    with st.expander("💪 生活方式 & 健康指标", expanded=False):
        stress = st.slider("压力评分 (1-10)", 1, 10, int(default_vals.get('stress_score', 5)))
        cognitive = st.slider("认知表现 (0-100)", 0, 100, int(default_vals.get('cognitive_performance_score', 75)))
        heart_rate = st.slider("静息心率 (bpm)", 40, 120, int(default_vals.get('heart_rate_resting_bpm', 70)))
        screen_time = st.number_input("睡前屏幕时间 (分钟)", 0, 300, int(default_vals.get('screen_time_before_bed_mins', 60)))
        bmi = st.number_input("BMI", 15.0, 45.0, float(default_vals.get('bmi', 22.0)), 0.1)
        steps = st.number_input("当日步数", 0, 20000, int(default_vals.get('steps_that_day', 6000)))
        work_hours = st.slider("工作时长 (小时)", 0, 16, int(default_vals.get('work_hours_that_day', 8)))
        if 'exercise_day' in num_cols:
            exercise = st.radio("当天是否锻炼", [0, 1], format_func=lambda x: "✅ 是" if x else "❌ 否", index=int(default_vals.get('exercise_day', 0)))
        else:
            exercise = 0
        if 'aid' in num_cols:
            aid = st.radio("使用助眠工具", [0, 1], format_func=lambda x: "✅ 是" if x else "❌ 否", index=int(default_vals.get('aid', 0)))
        else:
            aid = 0
        if 'shift' in num_cols:
            shift = st.radio("轮班工作", [0, 1], format_func=lambda x: "✅ 是" if x else "❌ 否", index=int(default_vals.get('shift', 0)))
        else:
            shift = 0
    with st.expander("🌡️ 环境及其他", expanded=False):
        chronotype = st.selectbox("睡眠类型", encoders['chronotype'].classes_)
        mental = st.selectbox("心理健康状况", encoders['mental_health_condition'].classes_)
        season = st.selectbox("季节", encoders['season'].classes_)
        day_type = st.selectbox("日期类型", encoders['day_type'].classes_)
        temp = st.slider("卧室温度 (°C)", 15, 30, int(default_vals.get('room_temperature_celsius', 22)))
        weekend_diff = st.slider("周末睡眠差 (小时)", -3.0, 5.0, float(default_vals.get('weekend_sleep_diff_hrs', 0.0)), 0.5)
        nap = st.number_input("午睡时长 (分钟)", 0, 120, int(default_vals.get('nap_duration_mins', 10)))
        caffeine = st.number_input("睡前咖啡因 (mg)", 0, 400, int(default_vals.get('caffeine_mg_before_bed', 50)))
        alcohol = st.number_input("睡前酒精 (单位)", 0, 5, int(default_vals.get('alcohol_units_before_bed', 0)))

# ==================== 构建用户输入特征 ====================
user_data = {}
for col in num_cols:
    if col in locals():
        user_data[col] = locals()[col]
    else:
        user_data[col] = default_vals.get(col, 0)

user_data['gender_encoded'] = encoders['gender'].transform([gender])[0]
user_data['occupation_encoded'] = encoders['occupation'].transform([occupation])[0]
user_data['country_encoded'] = encoders['country'].transform([country])[0]
user_data['chronotype_encoded'] = encoders['chronotype'].transform([chronotype])[0]
user_data['mental_health_condition_encoded'] = encoders['mental_health_condition'].transform([mental])[0]
user_data['season_encoded'] = encoders['season'].transform([season])[0]
user_data['day_type_encoded'] = encoders['day_type'].transform([day_type])[0]

input_df = pd.DataFrame([user_data])[feature_cols]
input_scaled = scaler.transform(input_df)

# ==================== 模型预测 ====================
model, model_acc, model_auc, model_cm, X_test, y_test, y_proba = train_model(model_choice, use_smote)

if hasattr(model, "predict_proba"):
    pred_proba = model.predict_proba(input_scaled)[0][1]
elif hasattr(model, "decision_function"):
    decision = model.decision_function(input_scaled)[0]
    pred_proba = 1 / (1 + np.exp(-decision))
else:
    pred_proba = 0.5
pred_class = int(pred_proba >= 0.5)

def get_sleep_score():
    quality_norm = sleep_quality / 10
    duration_norm = min(sleep_duration / 8, 1) if sleep_duration <= 8 else max(1 - (sleep_duration - 8) / 4, 0)
    rem_norm = min(rem / 25, 1) if rem <= 25 else max(1 - (rem - 25) / 15, 0)
    deep_norm = min(deep / 20, 1) if deep <= 20 else max(1 - (deep - 20) / 15, 0)
    score = (quality_norm * 0.4 + duration_norm * 0.3 + rem_norm * 0.15 + deep_norm * 0.15) * 100
    return round(score, 1)

sleep_score = get_sleep_score()

# ==================== 动态主内容 ====================
st.markdown('<div class="main-container" style="padding: 0 2rem 2rem 2rem;">', unsafe_allow_html=True)

if st.session_state.current_page == "仪表板":
    # ---------- 原有仪表板内容（完全不变） ----------
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
        <div>
            <span style="font-size: 0.9rem; background: #eef2fa; padding: 0.2rem 1rem; border-radius: 40px;">今日分析</span>
            <h2 style="margin: 0.5rem 0 0 0;">您好，{gender} · {age}岁</h2>
        </div>
        <div style="font-size: 0.85rem; color: #5c7f9c;">📅 {datetime.now().strftime("%Y年%m月%d日 %A")}</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="elegant-card" style="text-align: center;">
            <div style="font-size: 0.8rem; color:#5b7c9e;">睡眠障碍风险</div>
            <div style="font-size: 2.6rem; font-weight: 800; color: {'#2e7d32' if pred_class == 0 else '#d32f2f'};">{pred_proba:.1%}</div>
            <div style="font-size: 0.7rem;">{model_choice} 预测</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="elegant-card" style="text-align: center;">
            <div style="font-size: 0.8rem; color:#5b7c9e;">综合睡眠评分</div>
            <div style="font-size: 2.6rem; font-weight: 800; color:#1a6bb5;">{sleep_score}</div>
            <div style="font-size: 0.7rem;">/100 · {'优秀' if sleep_score >= 80 else '待改善'}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="elegant-card" style="text-align: center;">
            <div style="font-size: 0.8rem; color:#5b7c9e;">睡眠时长</div>
            <div style="font-size: 2.6rem; font-weight: 800;">{sleep_duration}h</div>
            <div style="font-size: 0.7rem;">{'达标 ≥7h' if sleep_duration >= 7 else '不足 <7h'}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="elegant-card" style="text-align: center;">
            <div style="font-size: 0.8rem; color:#5b7c9e;">压力水平</div>
            <div style="font-size: 2.6rem; font-weight: 800;">{stress}/10</div>
            <div style="font-size: 0.7rem;">{'偏低' if stress <= 4 else '适中' if stress <= 7 else '偏高'}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_left2, col_right2 = st.columns([1, 1])
    with col_left2:
        with st.expander("🔬 交互式风险模拟器（改变习惯，实时反馈）", expanded=True):
            st.markdown("调整下方参数，观察风险变化曲线：")
            sim_duration = st.slider("模拟睡眠时长 (小时)", 3.0, 10.0, sleep_duration, 0.1, key="sim_dur_web")
            sim_stress = st.slider("模拟压力评分 (1-10)", 1, 10, stress, 1, key="sim_str_web")
            sim_caffeine = st.slider("模拟睡前咖啡因 (mg)", 0, 400, int(user_data.get('caffeine_mg_before_bed', 50)), 10)
            sim_data = user_data.copy()
            sim_data['sleep_duration_hrs'] = sim_duration
            sim_data['stress_score'] = sim_stress
            sim_data['caffeine_mg_before_bed'] = sim_caffeine
            sim_df = pd.DataFrame([sim_data])[feature_cols]
            sim_scaled = scaler.transform(sim_df)
            if hasattr(model, "predict_proba"):
                sim_proba = model.predict_proba(sim_scaled)[0][1]
            else:
                sim_proba = 1 / (1 + np.exp(-model.decision_function(sim_scaled)[0]))
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("当前风险概率", f"{pred_proba:.1%}")
            col_s2.metric("模拟风险概率", f"{sim_proba:.1%}")
            delta_val = sim_proba - pred_proba
            col_s3.metric("变化趋势", f"{delta_val:+.1%}", delta=f"{abs(delta_val):.1%}" if delta_val != 0 else "0%", delta_color="inverse")
            if sim_proba < pred_proba:
                st.success("✅ 调整后风险降低！建议坚持这些好习惯。")
            else:
                st.warning("⚠️ 模拟后风险升高，改善睡眠时长或管理压力/咖啡因摄入有益。")

    with col_right2:
        radar_metrics = {
            "睡眠质量": sleep_quality / 10,
            "睡眠时长": min(sleep_duration / 8, 1.2),
            "压力(逆)": 1 - (stress - 1) / 9,
            "心率健康": 1 - max(0, min(1, (heart_rate - 60) / 40)),
            "深睡比例": deep / 35
        }
        radar_df = pd.DataFrame(dict(r=list(radar_metrics.values()), theta=list(radar_metrics.keys())))
        fig_radar = px.line_polar(radar_df, r='r', theta='theta', line_close=True, range_r=[0, 1],
                                  title="核心健康维度雷达图 (越高越好)", template="plotly_white")
        fig_radar.update_traces(fill='toself', fillcolor='rgba(26,107,181,0.2)', line_color='#1a6bb5')
        fig_radar.update_layout(height=340, margin=dict(l=40, r=40, t=60, b=40))
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_fact, col_advice = st.columns([1, 1])
    with col_fact:
        st.markdown("#### 🚨 风险因素检测")
        risk_factors = []
        if sleep_quality < 6: risk_factors.append(("睡眠质量偏低", sleep_quality, 6))
        if stress > 7: risk_factors.append(("压力偏高", stress, 7))
        if sleep_duration < 6: risk_factors.append(("睡眠不足", sleep_duration, 6))
        if screen_time > 120: risk_factors.append(("睡前屏幕时间过长", screen_time, 120))
        if heart_rate > 85: risk_factors.append(("静息心率偏高", heart_rate, 85))
        if cognitive < 60: risk_factors.append(("认知表现偏低", cognitive, 60))
        if rem < 15: risk_factors.append(("REM睡眠不足", rem, 15))
        if deep < 12: risk_factors.append(("深睡不足", deep, 12))
        if risk_factors:
            for factor, val, threshold in risk_factors:
                ratio = min(val / threshold, 1) if "不足" in factor or "偏低" in factor else min(threshold / val, 1)
                st.markdown(f"""
                <div style="margin-bottom:1rem;"><div style="display:flex; justify-content:space-between; font-size:0.85rem;">
                <span>⚠️ {factor}</span><span>{val} / {threshold}</span></div>
                <div class="custom-progress"><div class="custom-progress-bar" style="width:{ratio * 100}%; background:#ef4444;"></div></div></div>
                """, unsafe_allow_html=True)
        else:
            st.success("✨ 所有指标均在理想范围内，睡眠健康良好。")

    with col_advice:
        st.markdown("#### 💡 个性化建议 (基于循证)")
        suggestions = [
            ("🕒 固定作息", "保证7-8小时睡眠，每天同一时间上床和起床", sleep_duration < 7 or sleep_quality < 6),
            ("📵 睡前远离电子设备", "睡前1小时关闭所有屏幕，尝试冥想或阅读", sleep_quality < 6 or screen_time > 90),
            ("🧘 正念呼吸", "每天10分钟深呼吸练习，有效降低压力", stress > 7),
            ("🔵 蓝光过滤", "睡前开启蓝光过滤或佩戴防蓝光眼镜", screen_time > 90),
            ("🏃 有氧运动", "每周150分钟中等强度运动（快走、游泳）", heart_rate > 85 or stress > 6)
        ]
        for title, desc, condition in suggestions:
            if condition:
                st.markdown(f"""<div class="suggestion-card"><strong>{title}</strong><br>{desc}</div>""", unsafe_allow_html=True)
        if not any([s[2] for s in suggestions]):
            st.success("🌟 继续保持良好习惯，您的睡眠健康非常棒！")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if "history" not in st.session_state:
        st.session_state.history = []
    his_col, dl_col = st.columns(2)
    with his_col:
        if st.button("📌 保存本次评估结果", use_container_width=True):
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "model": model_choice,
                "risk": pred_proba,
                "sleep_score": sleep_score,
                "sleep_quality": sleep_quality,
                "sleep_duration": sleep_duration,
                "stress": stress
            }
            st.session_state.history.append(record)
            st.success("已保存当前记录")
    with dl_col:
        report_md = f"""
# SleepWell AI 睡眠健康报告
**生成时间**：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**使用模型**：{model_choice} {'(SMOTE)' if use_smote else ''}

## 评估摘要
- **睡眠障碍风险概率**：{pred_proba:.1%}
- **睡眠综合评分**：{sleep_score}/100
- **风险等级**：{'✅ 低风险' if pred_class == 0 else '⚠️ 高风险'}

## 关键指标
- 睡眠质量：{sleep_quality}/10
- 睡眠时长：{sleep_duration}小时
- 压力评分：{stress}/10
- 静息心率：{heart_rate} bpm
- REM睡眠：{rem}%
- 深睡：{deep}%

## 个性化建议
{chr(10).join([f"- {s[0]}: {s[1]}" for s in suggestions if s[2]]) if any([s[2] for s in suggestions]) else "- 继续保持良好习惯"}

---
*本报告由 AI 生成，仅供参考。*
"""
        st.download_button(label="📥 导出健康报告 (Markdown)", data=report_md,
                           file_name=f"sleepwell_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md", mime="text/markdown",
                           use_container_width=True)

    if st.session_state.history:
        with st.expander("📊 历史评估记录对比 (追踪睡眠趋势)", expanded=False):
            hist_df = pd.DataFrame(st.session_state.history)
            st.dataframe(hist_df, use_container_width=True)
            if len(hist_df) > 1:
                fig_trend = px.line(hist_df, x="timestamp", y="risk", title="风险概率变化趋势", markers=True,
                                    color_discrete_sequence=["#1a6bb5"])
                fig_trend.update_layout(height=300)
                st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.caption("点击上方「保存本次评估结果」即可建立个人睡眠健康档案。")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 特征重要性 & 模型系数", "🎯 模型性能矩阵", "🧩 睡眠模式聚类 (健康人群)", "📊 与健康中位数对比"])
    with tab1:
        if hasattr(model, "feature_importances_"):
            imp_df = pd.DataFrame({'特征': feature_cols, '重要性': model.feature_importances_}).sort_values('重要性', ascending=True).tail(15)
            fig = px.bar(imp_df, x='重要性', y='特征', orientation='h', color='重要性', color_continuous_scale='Blues', title="Top 15 特征重要性 (随机森林)")
            fig.update_layout(height=500, margin=dict(l=0, r=0, t=60, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            if model_choice == "逻辑回归":
                coef_df = pd.DataFrame({'特征': feature_cols, '系数': model.coef_[0]}).sort_values('系数', key=abs, ascending=False).head(15)
                fig = px.bar(coef_df, x='系数', y='特征', orientation='h', title="逻辑回归系数 (绝对值越大影响越大)", color='系数', color_continuous_scale='RdBu')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("当前模型不支持特征重要性展示，请切换随机森林或逻辑回归查看。")
    with tab2:
        st.markdown(f"#### 当前模型：{model_choice} {'(SMOTE enabled)' if use_smote else ''}")
        col1, col2, col3 = st.columns(3)
        col1.metric("准确率 (Accuracy)", f"{model_acc:.3f}")
        col2.metric("AUC", f"{model_auc:.3f}" if model_auc else "N/A")
        f1 = 2 * model_cm[1, 1] / (2 * model_cm[1, 1] + model_cm[0, 1] + model_cm[1, 0]) if model_cm.shape == (2, 2) else 0
        col3.metric("F1分数 (风险类)", f"{f1:.3f}")
        col_cm, col_roc = st.columns(2)
        with col_cm:
            fig_cm = go.Figure(data=go.Heatmap(z=model_cm, x=['预测健康', '预测障碍'], y=['真实健康', '真实障碍'], text=model_cm, texttemplate="%{text}", colorscale='Blues'))
            fig_cm.update_layout(title="混淆矩阵", height=400)
            st.plotly_chart(fig_cm, use_container_width=True)
        with col_roc:
            if y_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{model_choice} (AUC={model_auc:.3f})', line=dict(color='#1e88e5', width=3)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='随机猜测', line=dict(dash='dash')))
                fig_roc.update_layout(title="ROC曲线", height=400)
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("当前模型无法生成ROC曲线（无概率输出）")
    with tab3:
        st.markdown("#### 🧩 基于健康人群的睡眠模式聚类 (无监督学习)")
        user_cluster, cluster_desc = predict_user_cluster(input_scaled.flatten())
        st.success(f"根据您的睡眠指标，您属于 **{cluster_desc}**")
        healthy_cluster_labels = kmeans_model.predict(pca_model.transform(healthy_df[feature_cols].values))
        cluster_counts = pd.Series(healthy_cluster_labels).value_counts().sort_index()
        fig_pie = px.pie(values=cluster_counts.values, names=[f"类型{i+1}" for i in cluster_counts.index], title="健康人群睡眠模式分布", hole=0.4, color_discrete_sequence=px.colors.sequential.Blues_r)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("""
        **聚类解读**：
        - 类型1（优质睡眠型）：睡眠质量高、压力低，深睡充足。
        - 类型2（普通睡眠型）：指标中等，仍有优化空间。
        - 类型3（需关注型）：睡眠时长或质量偏低，建议干预。
        """)
        if user_cluster == 0:
            st.info("🎉 您属于优质睡眠模式！继续保持规律作息和压力管理。")
        elif user_cluster == 1:
            st.info("📈 您属于普通睡眠模式，可以适当增加运动或减少咖啡因摄入。")
        else:
            st.warning("⚠️ 您属于需关注型，建议针对睡眠习惯做出改变，例如固定就寝时间。")
    with tab4:
        compare_keys = ['sleep_duration_hrs', 'sleep_quality_score', 'stress_score', 'heart_rate_resting_bpm', 'rem_percentage', 'deep_sleep_percentage']
        comp_data = []
        for key in compare_keys:
            if key in user_data:
                user_val = user_data[key]
                ref_val = healthy_df[key].median()
                diff = user_val - ref_val
                direction = "↑ 偏高" if diff > 0 else "↓ 偏低"
                good = (key in ['stress_score', 'heart_rate_resting_bpm'] and diff < 0) or (key not in ['stress_score', 'heart_rate_resting_bpm'] and diff > 0)
                comp_data.append({"指标": key, "您的值": user_val, "健康中位数": ref_val, "差异": f"{direction} {abs(diff):.1f}", "状态": "✓ 良好" if good else "⚠️ 注意"})
        df_comp = pd.DataFrame(comp_data)
        df_comp['偏离比'] = (df_comp['您的值'] / df_comp['健康中位数'] - 1) * 100
        fig_comp = px.bar(df_comp, x='偏离比', y='指标', orientation='h', color='状态', color_discrete_map={'✓ 良好': '#2e7d32', '⚠️ 注意': '#d32f2f'}, title="与健康人群中位数偏差 (%)")
        fig_comp.update_layout(height=400)
        st.plotly_chart(fig_comp, use_container_width=True)

    with st.expander("⚙️ 系统架构与模型参数", expanded=False):
        st.markdown(f"""
        - **当前选用模型**：{model_choice}  
        - **SMOTE样本平衡**：{'已启用' if use_smote else '未启用'}  
        - **特征总数**：{len(feature_cols)}，**训练样本量**：{len(df_processed)}  
        - **健康/高风险分布**：{(df_processed['risk_label'] == 0).mean():.1%} / {(df_processed['risk_label'] == 1).mean():.1%}
        - **聚类簇数**：{n_clusters} (基于健康人群)
        """)

elif st.session_state.current_page == "分析报告":
    st.markdown("## 📈 分析报告")
    st.markdown("基于您当前的健康档案与历史趋势，AI 为您生成以下深度分析报告。")
    st.markdown("---")

    # ---------- 1. 综合风险评级卡片 ----------
    risk_color = "#2e7d32" if pred_class == 0 else "#d32f2f"
    risk_level = "低风险 · 睡眠健康" if pred_class == 0 else "高风险 · 建议干预"
    st.markdown(f"""
    <div class="elegant-card" style="margin-bottom: 1.5rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 style="margin:0;">综合睡眠障碍风险评级</h3>
                <p style="margin:0.2rem 0 0; color: #5b7c9e;">基于 {model_choice} 模型预测</p>
            </div>
            <div style="text-align: right;">
                <span style="background: {risk_color}10; color: {risk_color}; padding:0.3rem 1rem; border-radius:40px; font-weight:600;">{risk_level}</span>
                <div style="font-size: 2.5rem; font-weight: 800; margin-top: 0.2rem;">{pred_proba:.1%}</div>
            </div>
        </div>
        <div class="custom-progress" style="margin-top: 0.8rem;">
            <div class="custom-progress-bar" style="width: {pred_proba*100}%; background: {risk_color};"></div>
        </div>
        <p style="margin-top: 0.8rem; font-size:0.85rem;">
            {"✅ 当前睡眠习惯良好，请继续保持并定期监测。" if pred_class==0 else "⚠️ 存在较高的睡眠障碍风险，建议重点关注下方风险因素并采取改善措施。"}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ---------- 2. 关键指标卡片（四列） ----------
    kpi_cols = st.columns(4)
    metrics = [
        ("睡眠质量", f"{sleep_quality}/10", "良好" if sleep_quality>=6 else "需改善"),
        ("睡眠时长", f"{sleep_duration}h", "达标" if sleep_duration>=7 else "不足"),
        ("压力评分", f"{stress}/10", "正常" if stress<=5 else "偏高"),
        ("静息心率", f"{heart_rate} bpm", "正常" if 60<=heart_rate<=85 else "异常")
    ]
    for i, (label, val, status) in enumerate(metrics):
        with kpi_cols[i]:
            st.markdown(f"""
            <div class="metric-tile">
                <div style="font-size:0.8rem; color:#5b7c9e;">{label}</div>
                <div style="font-size:1.8rem; font-weight:700;">{val}</div>
                <div style="font-size:0.7rem; color: {'#2e7d32' if '良好' in status or '达标' in status or '正常' in status else '#d32f2f'};">{status}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ---------- 3. 深度指标雷达图（与健康人群对比） ----------
    st.markdown("#### 🧠 与健康人群对比分析")
    # 选取几个核心指标进行比较
    compare_keys = ['sleep_quality_score', 'sleep_duration_hrs', 'stress_score', 'heart_rate_resting_bpm', 'rem_percentage', 'deep_sleep_percentage']
    user_values = []
    healthy_medians = []
    for key in compare_keys:
        if key in user_data:
            user_values.append(user_data[key])
        else:
            user_values.append(0)
        healthy_medians.append(healthy_df[key].median())
    # 归一化处理 (将stress和心率反向，使得"越高越好")
    norm_user = []
    norm_healthy = []
    for i, key in enumerate(compare_keys):
        if key in ['stress_score', 'heart_rate_resting_bpm']:
            # 这些指标越低越好，取倒数归一化（限制范围）
            max_val = max(healthy_df[key].max(), user_data.get(key, healthy_df[key].max()))
            min_val = min(healthy_df[key].min(), user_data.get(key, healthy_df[key].min()))
            if max_val == min_val:
                norm_u = 0.5
                norm_h = 0.5
            else:
                norm_u = 1 - (user_values[i] - min_val) / (max_val - min_val)
                norm_h = 1 - (healthy_medians[i] - min_val) / (max_val - min_val)
        else:
            max_val = max(healthy_df[key].max(), user_data.get(key, healthy_df[key].max()))
            min_val = min(healthy_df[key].min(), user_data.get(key, healthy_df[key].min()))
            if max_val == min_val:
                norm_u = 0.5
                norm_h = 0.5
            else:
                norm_u = (user_values[i] - min_val) / (max_val - min_val)
                norm_h = (healthy_medians[i] - min_val) / (max_val - min_val)
        norm_user.append(norm_u)
        norm_healthy.append(norm_h)

    radar_df = pd.DataFrame({
        '指标': ['睡眠质量', '睡眠时长', '压力(逆)', '静息心率(逆)', 'REM占比', '深睡占比'],
        '您的水平': norm_user,
        '健康人群中位数': norm_healthy
    })
    fig_radar = px.line_polar(radar_df, r='您的水平', theta='指标', line_close=True, range_r=[0,1],
                              title="核心指标雷达图 (越靠近外圈越优)", template="plotly_white")
    fig_radar.add_trace(go.Scatterpolar(r=radar_df['健康人群中位数'], theta=radar_df['指标'], fill='toself',
                                        name='健康人群中位数', line=dict(dash='dash', color='#2e7d32'),
                                        fillcolor='rgba(46,125,50,0.1)'))
    fig_radar.update_layout(height=450)
    st.plotly_chart(fig_radar, use_container_width=True)
    st.caption("说明：压力与心率已做逆向处理，数值越高代表越健康。您的数据与健康人群对比可直观发现短板。")

    st.markdown("---")

    # ---------- 4. 风险因素详解（表格形式） ----------
    st.markdown("#### ⚠️ 主要风险因素识别")
    risk_items = []
    if sleep_quality < 6: risk_items.append(("睡眠质量偏低", f"{sleep_quality}/10", "建议改善睡眠环境、固定作息"))
    if stress > 7: risk_items.append(("压力水平偏高", f"{stress}/10", "尝试冥想、深呼吸或心理疏导"))
    if sleep_duration < 6: risk_items.append(("睡眠时长不足", f"{sleep_duration}h", "保证7-8小时睡眠，避免熬夜"))
    if screen_time > 120: risk_items.append(("睡前屏幕时间过长", f"{screen_time}分钟", "睡前1小时禁用电子设备"))
    if heart_rate > 85: risk_items.append(("静息心率偏高", f"{heart_rate} bpm", "增加有氧运动，减少咖啡因"))
    if cognitive < 60: risk_items.append(("认知表现偏低", f"{cognitive}/100", "改善睡眠质量，补充B族维生素"))
    if rem < 15: risk_items.append(("REM睡眠不足", f"{rem}%", "避免酒精，增加规律睡眠"))
    if deep < 12: risk_items.append(("深睡不足", f"{deep}%", "降低睡前压力，保持卧室凉爽"))

    if risk_items:
        risk_df = pd.DataFrame(risk_items, columns=["风险项", "当前值", "改善建议"])
        st.dataframe(risk_df, use_container_width=True, hide_index=True)
    else:
        st.success("✅ 所有核心指标均在理想范围，无明显风险因素。")

    st.markdown("---")

    # ---------- 5. 历史趋势分析（如果有历史记录） ----------
    st.markdown("#### 📉 历史风险趋势")
    if st.session_state.get("history") and len(st.session_state.history) > 1:
        hist_df = pd.DataFrame(st.session_state.history)
        hist_df['timestamp_dt'] = pd.to_datetime(hist_df['timestamp'])
        hist_df = hist_df.sort_values('timestamp_dt')
        fig_trend = px.line(hist_df, x='timestamp', y='risk', title="睡眠障碍风险概率变化趋势",
                            markers=True, color_discrete_sequence=["#d32f2f"])
        fig_trend.update_layout(height=350)
        st.plotly_chart(fig_trend, use_container_width=True)
        # 简单趋势评价
        latest_risk = hist_df['risk'].iloc[-1]
        first_risk = hist_df['risk'].iloc[0]
        if latest_risk < first_risk:
            st.success("📉 相比首次记录，您的风险概率呈下降趋势，改善措施有效！")
        elif latest_risk > first_risk:
            st.warning("📈 风险概率较首次记录有所上升，请回顾近期生活习惯并调整。")
        else:
            st.info("风险概率保持稳定，继续维持良好习惯。")
    else:
        st.info("暂无足够的历史数据。请使用仪表板中的「保存本次评估结果」按钮记录多次测量，即可在此查看趋势分析。")

    st.markdown("---")

    # ---------- 6. 个性化改善计划（可打印报告） ----------
    st.markdown("#### 🎯 专属改善计划 (基于您的数据)")
    # 收集建议条目
    suggestions = []
    if sleep_duration < 7 or sleep_quality < 6:
        suggestions.append("🕒固定作息：每天同一时间上床和起床，周末也不例外。")
    if stress > 7:
        suggestions.append("🧘压力管理：每天进行10分钟正念呼吸或渐进式肌肉放松。")
    if screen_time > 90:
        suggestions.append("📵数字排毒：睡前一小时关闭所有电子屏幕，改为阅读或听轻音乐。")
    if heart_rate > 85:
        suggestions.append("🏃有氧运动：每周至少150分钟中等强度运动（快走、游泳、骑行）。")
    if rem < 15 or deep < 12:
        suggestions.append("🌡️优化睡眠环境：保持卧室温度18-22℃，使用遮光窗帘，减少噪音。")
    if caffeine > 200:
        suggestions.append("☕减少咖啡因：下午2点后避免咖啡、浓茶或能量饮料。")
    if alcohol > 2:
        suggestions.append("🍷限制酒精：酒精会破坏REM睡眠，建议睡前4小时不饮酒。")
    if not suggestions:
        suggestions.append("🌟您已拥有非常健康的睡眠模式，继续保持！建议每月定期评估一次。")

    for s in suggestions:
        st.markdown(f"<div class='suggestion-card' style='margin-bottom:0.6rem;'>{s}</div>", unsafe_allow_html=True)

    # ---------- 7. 一键导出完整报告（增强版） ----------
    st.markdown("#### 📄 导出报告")
    # 生成包含所有图表和数据的详细报告HTML（供下载）
    report_html = f"""
    <html>
    <head><meta charset="UTF-8"><title>SleepWell AI 健康报告</title>
    <style>body {{ font-family: 'Inter', sans-serif; padding: 2rem; }} .card {{ border:1px solid #ddd; border-radius: 20px; padding:1rem; margin-bottom:1rem; }}</style>
    </head>
    <body>
    <h1>🌙 SleepWell AI 睡眠健康报告</h1>
    <p><strong>生成时间：</strong>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    <p><strong>使用模型：</strong>{model_choice} {'(SMOTE)' if use_smote else ''}</p>
    <hr>
    <h2>📊 评估摘要</h2>
    <ul>
        <li>睡眠障碍风险概率：<strong>{pred_proba:.1%}</strong> ({risk_level})</li>
        <li>综合睡眠评分：{sleep_score}/100</li>
        <li>睡眠质量：{sleep_quality}/10 &nbsp; 睡眠时长：{sleep_duration}h &nbsp; 压力评分：{stress}/10 &nbsp; 静息心率：{heart_rate} bpm</li>
    </ul>
    <hr>
    <h2>⚠️ 风险因素</h2>
    <ul>
    {"".join([f"<li><strong>{r[0]}</strong>：{r[1]}（建议：{r[2]}）</li>" for r in risk_items]) if risk_items else "<li>无显著风险</li>"}
    </ul>
    <hr>
    <h2>💡 改善计划</h2>
    <ul>
    {"".join([f"<li>{s}</li>" for s in suggestions])}
    </ul>
    <hr>
    <p style="font-size:0.8rem;">⚠️ 免责声明：本报告由AI生成，仅供参考，不构成医疗诊断。如有严重睡眠问题请咨询专业医生。</p>
    </body>
    </html>
    """
    st.download_button("📥 下载完整报告 (HTML)", data=report_html, file_name=f"sleepwell_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                       mime="text/html", use_container_width=True)

    st.markdown("""
    <div style="background:#f9fafc; border-radius:20px; padding:1rem; margin-top:1rem; font-size:0.75rem; color:#6c86a0; text-align:center;">
        本报告基于您最近一次侧边栏输入的数据生成。如需更新数据，请修改左侧健康档案并返回仪表板重新保存。
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.current_page == "健康指南":
    st.markdown("## 💚 健康指南")
    st.markdown("基于您的睡眠数据，AI 为您定制了以下改善方案。")
    st.markdown("---")

    # ---------- 1. 当前状态卡片 ----------
    col_status, col_score = st.columns(2)
    with col_status:
        status_color = "#2e7d32" if pred_class == 0 else "#d32f2f"
        status_text = "睡眠健康状态良好" if pred_class == 0 else "存在睡眠风险，建议积极干预"
        st.markdown(f"""
        <div class="elegant-card" style="text-align: center;">
            <div style="font-size: 1.2rem; font-weight: 600; color: {status_color};">{status_text}</div>
            <div style="margin-top: 0.5rem;">综合睡眠评分 <strong>{sleep_score}</strong> / 100</div>
            <div style="font-size: 0.8rem; color: #5b7c9e;">基于 {model_choice} 预测</div>
        </div>
        """, unsafe_allow_html=True)

    with col_score:
        st.markdown(f"""
        <div class="elegant-card" style="text-align: center;">
            <div style="font-size: 1rem; color:#5b7c9e;">核心指标达标情况</div>
            <div style="margin-top: 0.5rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
                    <span>睡眠时长 ≥7h</span>
                    <span style="color: {'#2e7d32' if sleep_duration >= 7 else '#d32f2f'};">{'✅' if sleep_duration >= 7 else '❌'}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
                    <span>睡眠质量 ≥6分</span>
                    <span style="color: {'#2e7d32' if sleep_quality >= 6 else '#d32f2f'};">{'✅' if sleep_quality >= 6 else '❌'}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
                    <span>压力 ≤5分</span>
                    <span style="color: {'#2e7d32' if stress <= 5 else '#d32f2f'};">{'✅' if stress <= 5 else '❌'}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>心率 60-85 bpm</span>
                    <span style="color: {'#2e7d32' if 60 <= heart_rate <= 85 else '#d32f2f'};">{'✅' if 60 <= heart_rate <= 85 else '❌'}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ---------- 2. 个性化改善建议 ----------
    st.markdown("#### 🎯 您的专属改善计划")

    improvement_areas = []
    if sleep_duration < 7:
        improvement_areas.append(("⏰ 睡眠时长不足", f"当前 {sleep_duration}h，建议达到 7-8h",
                                   ["固定就寝与起床时间（包括周末）", "午睡不超过30分钟", "避免睡前剧烈运动"]))
    if sleep_quality < 6:
        improvement_areas.append(("😴 睡眠质量偏低", f"当前 {sleep_quality}/10",
                                   ["睡前1小时停止使用电子设备", "尝试白噪音或轻音乐", "保持卧室黑暗、凉爽"]))
    if stress > 7:
        improvement_areas.append(("😫 压力水平过高", f"当前 {stress}/10",
                                   ["每天10分钟正念呼吸", "写感恩日记或情绪笔记", "每周2-3次中等强度运动"]))
    if heart_rate > 85:
        improvement_areas.append(("💓 静息心率偏高", f"当前 {heart_rate} bpm",
                                   ["增加有氧运动（快走、游泳）", "减少咖啡因摄入", "保证充足水分"]))
    if screen_time > 90:
        improvement_areas.append(("📱 睡前屏幕时间过长", f"当前 {screen_time} 分钟",
                                   ["睡前60分钟开启夜间模式", "使用物理闹钟替代手机", "养成阅读纸质书习惯"]))
    if caffeine > 200:
        improvement_areas.append(("☕ 咖啡因摄入过多", f"当前 {caffeine} mg/天",
                                   ["下午2点后不喝咖啡/浓茶", "改用低因咖啡或草本茶", "记录每日咖啡因来源"]))
    if alcohol > 2:
        improvement_areas.append(("🍷 酒精影响睡眠", f"当前 {alcohol} 单位/天",
                                   ["睡前4小时不饮酒", "用无酒精饮品替代", "了解酒精会破坏REM睡眠"]))
    if exercise == 0:
        improvement_areas.append(("🏃 缺乏运动", "当天未记录锻炼",
                                   ["每周至少150分钟中等强度运动", "从每天散步20分钟开始", "尝试早晨运动以调节生物钟"]))
    if rem < 15:
        improvement_areas.append(("💤 REM睡眠不足", f"当前 {rem}%",
                                   ["规律作息，避免酒精", "尝试梦境回顾练习", "增加维生素B6摄入"]))
    if deep < 12:
        improvement_areas.append(("🌙 深睡不足", f"当前 {deep}%",
                                   ["降低睡前压力", "睡前泡热水澡", "使用重力毯或白噪音"]))
    if not improvement_areas:
        improvement_areas.append(("🌟 保持优异", "所有指标均很理想", ["继续保持现有习惯", "每月定期自评一次", "分享健康经验给亲友"]))

    # 展示改善区域（已修复重复图标问题）
    for area, desc, tips in improvement_areas:
        with st.container():
            st.markdown(f"""
            <div class="elegant-card" style="margin-bottom: 1rem;">
                <div>
                    <div style="font-weight: 700; font-size: 1.1rem;">{area}</div>
                    <div style="font-size: 0.85rem; color: #5b7c9e;">{desc}</div>
                </div>
                <div style="margin-top: 0.8rem; padding-left: 1rem;">
                    {"".join([f"<div style='margin-bottom: 0.3rem;'>• {tip}</div>" for tip in tips])}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ---------- 3. 一周行动计划 ----------
    st.markdown("#### 📅 一周睡眠改善行动计划")
    plan = []
    if sleep_duration < 7:
        plan.append("周一：设定固定的上床和起床时间，今晚提前30分钟上床。")
        plan.append("周三：记录一周睡眠日志，观察趋势。")
    if stress > 7:
        plan.append("每天：早晨10分钟正念呼吸，睡前写感恩日记。")
        plan.append("周五：安排一次轻度瑜伽或拉伸放松。")
    if screen_time > 90:
        plan.append("每晚：20:00后改为阅读或听播客，手机放客厅充电。")
    if exercise == 0:
        plan.append("周二/周四：快走30分钟，使用计步器目标8000步。")
    if not plan:
        plan.append("本周：继续保持现有习惯，可以尝试一种新的放松方式（如冥想、泡澡）。")
        plan.append("周末：与家人朋友分享您的睡眠健康经验。")

    plan.append("每日：保持卧室温度18-22℃，使用遮光窗帘。")
    plan.append("周末：允许自己一次自然醒，但差异不超过1小时。")

    for item in plan[:6]:
        st.markdown(f"<div class='suggestion-card' style='margin-bottom: 0.6rem;'>{item}</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ---------- 4. 科学小贴士 ----------
    st.markdown("#### 💡 科学睡眠小贴士")
    tips_dict = {
        "睡眠时长": "成年人需要7-9小时睡眠，长期少于6小时会增加心血管疾病风险。",
        "压力": "压力激素皮质醇升高会抑制褪黑素分泌，睡前1小时避免思考工作。",
        "屏幕时间": "屏幕蓝光会抑制褪黑素达50%，使用防蓝光眼镜或开启夜间模式。",
        "咖啡因": "咖啡因半衰期约5小时，下午2点后摄入会影响夜间深睡。",
        "酒精": "酒精虽能助眠，但会大幅减少REM睡眠，导致后半夜易醒。",
        "运动": "早晨或下午运动效果最佳，睡前2小时避免剧烈运动。",
        "心率": "静息心率偏高可能提示压力或缺乏有氧训练，规律运动可改善。",
        "REM/深睡": "深睡主要在前半夜，REM睡眠在后半夜，早睡有助于增加深睡。",
    }
    shown = set()
    if sleep_duration < 7 and "睡眠时长" not in shown:
        st.info(tips_dict["睡眠时长"])
        shown.add("睡眠时长")
    if stress > 7 and "压力" not in shown:
        st.info(tips_dict["压力"])
        shown.add("压力")
    if screen_time > 90 and "屏幕时间" not in shown:
        st.info(tips_dict["屏幕时间"])
        shown.add("屏幕时间")
    if caffeine > 200 and "咖啡因" not in shown:
        st.info(tips_dict["咖啡因"])
        shown.add("咖啡因")
    if alcohol > 2 and "酒精" not in shown:
        st.info(tips_dict["酒精"])
        shown.add("酒精")
    if exercise == 0 and "运动" not in shown:
        st.info(tips_dict["运动"])
        shown.add("运动")
    if heart_rate > 85 and "心率" not in shown:
        st.info(tips_dict["心率"])
        shown.add("心率")
    if rem < 15 or deep < 12:
        st.info(tips_dict["REM/深睡"])
        shown.add("REM/深睡")
    if not shown:
        st.success("✨ 您已经拥有不错的睡眠习惯！继续坚持，也可以分享给朋友。")

    st.markdown("---")
    st.caption("提示：您的侧边栏数据会实时影响以上建议。如需更新，请修改左侧健康档案。")
elif st.session_state.current_page == "关于我们":
    st.markdown("## 🌟 关于 SleepWell AI")
    st.write("SleepWell AI 致力于用机器学习改善睡眠健康。")
    st.write("本工具基于真实睡眠健康数据集训练，提供多模型风险评估和个性化建议。")
    st.markdown("""
    - **数据来源**：睡眠健康公开数据集
    - **模型算法**：随机森林、逻辑回归、线性SVM
    - **核心技术**：SMOTE过采样、聚类分析、特征重要性
    """)
    st.caption("© 2025 SleepWell AI · 用科技改善睡眠健康")

elif st.session_state.current_page == "个人档案":
    st.markdown("## 👤 个人档案")
    st.write("您当前的健康档案配置如下：")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**年龄**：{age} 岁")
        st.write(f"**性别**：{gender}")
        st.write(f"**职业**：{occupation}")
        st.write(f"**国家/地区**：{country}")
        st.write(f"**睡眠类型**：{chronotype}")
        st.write(f"**心理健康**：{mental}")
    with col2:
        st.write(f"**睡眠质量**：{sleep_quality}/10")
        st.write(f"**睡眠时长**：{sleep_duration} 小时")
        st.write(f"**压力评分**：{stress}/10")
        st.write(f"**静息心率**：{heart_rate} bpm")
        st.write(f"**BMI**：{bmi}")
    if st.button("🗑️ 清除所有历史记录", use_container_width=True):
        if "history" in st.session_state:
            st.session_state.history = []
        st.success("历史记录已清除")
    st.caption("您的数据仅保存在当前浏览器会话中，不会上传至任何服务器。")

# ==================== 底部公共区域 ====================
st.markdown("""
<div class="footer" style="margin-top: 2rem; text-align: center; padding: 1rem 0; border-top: 1px solid #e2edf5; font-size: 0.7rem; color:#6c86a0;">
    <p>⚠️ 重要提示：本工具所有分析结果仅供参考，不能替代专业医疗诊断。如有严重睡眠问题请及时就医。</p>
    <p>© 2025 SleepWell AI · 用科技改善睡眠健康 · 版本 2.0</p>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)