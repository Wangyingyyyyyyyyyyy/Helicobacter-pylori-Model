import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 加载模型和标准化器
if 'model' not in st.session_state:
    st.session_state["model"] = joblib.load('model.pkl')  # 使用 XGBoost 的加载方法
    st.session_state["scaler"] = joblib.load('scaler.pkl')
model = st.session_state["model"]
scaler = st.session_state["scaler"]


# 设置页面背景
def set_background():
    page_bg_img = '''
    <style>
    .css-1nnpbs {width: 100vw}
    h1 {padding: 0.75rem 0px 0.75rem;margin-top: 2rem;box-shadow: 0px 3px 5px gray;}
    h2 {background-color: #B266FF;margin-top: 2vh;border-left: red solid 0.6vh}
    .css-1avcm0n {background: rgb(14, 17, 23, 0)}
    .css-18ni7ap {background: #B266FF;z-index:1;height:3rem}
    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    .css-z5fcl4 {padding: 0 1rem 1rem}
    .css-12ttj6m {box-shadow: 0.05rem 0.05rem 0.2rem 0.1rem rgb(192, 192, 192);margin:0 calc(20% + 0.5rem);}
    .css-1cbqeqj {text-align: center;}
    .css-1x8cf1d {background: #00800082}
    .css-1x8cf1d:hover {background: #00800033}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)


set_background()
st.markdown(
    "<h1 style='text-align: center'>Helicobacter pylori Multidrug Resistance Prediction Based on a Machine Learning Model</h1>",
    unsafe_allow_html=True)

# 创建表单
with st.form("my_form"):
    col7, col8 = st.columns([5, 5])
    with col7:
        a = st.selectbox("gyrA P116A", ("presence", "absence"))
        b = st.selectbox("group_505", ("presence", "absence"))
        c = st.selectbox("group_354", ("presence", "absence"))
        d = st.selectbox("glmU E162T", ("presence", "absence"))
        e = st.selectbox("HP0602 I67T", ("presence", "absence"))
        f = st.selectbox("group_1303", ("presence", "absence"))
    with col8:
        g = st.selectbox("group_541", ("presence", "absence"))
        h = st.selectbox("omp13 L29A", ("presence", "absence"))
        i = st.selectbox("ydjA", ("presence", "absence"))
        j = st.selectbox("group_283", ("presence", "absence"))
        k = st.selectbox("HP0757 G878A", ("presence", "absence"))
        l = st.selectbox("HP0486 A11T", ("presence", "absence"))

    col4, col5, col6 = st.columns([2, 2, 6])
    with col4:
        submitted = st.form_submit_button("Predict")
    with col5:
        reset = st.form_submit_button("Reset")

    if submitted:
        # 创建用户输入的数据框
        X = pd.DataFrame([[a, b, c, d, e, f, g, h, i, j, k, l]],
                         columns=['gyrA P116A', 'group_505', 'group_354', 'glmU E162T', 'HP0602 I67T', 'group_1303',
                                  'group_541', 'omp13 L29A', 'ydjA', 'group_283', 'HP0757 G878A', 'HP0486 A11T'])
        X = X.replace(["presence", "absence"], [1, 0])

        # 标准化用户输入数据
        X_scaled = scaler.transform(X)

        # 预测
        prediction = model.predict(X_scaled)[0]
        Predict_proba = model.predict_proba(X_scaled)[:, 1][0]

        if prediction == 0:
            st.subheader(f"The predicted result of Hp MDR: Sensitive")
        else:
            st.subheader(f"The predicted result of Hp MDR: Resistant")

        st.subheader(f"The probability of Hp MDR: {'%.2f' % float(Predict_proba * 100) + '%'}")

        with st.spinner('Please wait...'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
            shap.force_plot(explainer.expected_value, shap_values[0], X.iloc[0].values,
                            feature_names=['gyrA P116A', 'group_505', 'group_354', 'glmU E162T', 'HP0602 I67T',
                                           'group_1303',
                                           'group_541', 'omp13 L29A', 'ydjA', 'group_283', 'HP0757 G878A',
                                           'HP0486 A11T'], matplotlib=True, show=False, figsize=(20, 5))
            plt.xticks(fontproperties='Arial', size=16)
            plt.yticks(fontproperties='Arial', size=16)
            plt.tight_layout()
            plt.savefig('force.png', dpi=600)
            st.image('force.png')