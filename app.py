import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Risk Scorer", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load("credit_model.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    explainer = shap.TreeExplainer(
        model,
        model_output='raw',
        feature_perturbation='tree_path_dependent'
    )
    return model, feature_cols, explainer

model, feature_cols, explainer = load_model()

st.title("Credit Risk Scoring Model")
st.markdown("Enter a borrower's financial details to get a default risk score and explanation.")

st.sidebar.header("Borrower details")

age = st.sidebar.slider("Age", 18, 90, 35)
income = st.sidebar.number_input("Monthly income ($)", 0, 50000, 5000, step=500)
revolving_util = st.sidebar.slider("Revolving utilization (0–1)", 0.0, 1.0, 0.3, step=0.01)
debt_ratio = st.sidebar.slider("Debt ratio", 0.0, 1.0, 0.3, step=0.01)
open_credit = st.sidebar.slider("Open credit lines & loans", 0, 30, 5)
real_estate = st.sidebar.slider("Real estate loans", 0, 10, 1)
dependents = st.sidebar.slider("Number of dependents", 0, 10, 0)
late_30_59 = st.sidebar.slider("Times 30–59 days late", 0, 10, 0)
late_60_89 = st.sidebar.slider("Times 60–89 days late", 0, 10, 0)
late_90 = st.sidebar.slider("Times 90+ days late", 0, 10, 0)

total_late = late_30_59 + late_60_89 + late_90
debt_to_income = debt_ratio * income
income_per_dep = income / (dependents + 1)
income_missing = 0

input_data = pd.DataFrame([{
    'RevolvingUtilizationOfUnsecuredLines': revolving_util,
    'age': age,
    'NumberOfTime30-59DaysPastDueNotWorse': late_30_59,
    'DebtRatio': debt_ratio,
    'MonthlyIncome': income,
    'NumberOfOpenCreditLinesAndLoans': open_credit,
    'NumberOfTimes90DaysLate': late_90,
    'NumberRealEstateLoansOrLines': real_estate,
    'NumberOfTime60-89DaysPastDueNotWorse': late_60_89,
    'NumberOfDependents': dependents,
    'income_missing': income_missing,
    'debt_to_income': debt_to_income,
    'total_late_payments': total_late,
    'income_per_dependent': income_per_dep,
}])[feature_cols]

if st.sidebar.button("Calculate risk score", type="primary"):

    prob = model.predict_proba(input_data)[0][1]
    risk_pct = round(prob * 100, 1)

    col1, col2, col3 = st.columns(3)
    col1.metric("Default probability", f"{risk_pct}%")

    if prob < 0.2:
        col2.success("Low risk")
        verdict = "Low risk — likely to repay"
    elif prob < 0.5:
        col2.warning("Medium risk")
        verdict = "Medium risk — proceed with caution"
    else:
        col2.error("High risk")
        verdict = "High risk — likely to default"

    col3.metric("Model AUC", "0.85")
    st.markdown(f"**Assessment:** {verdict}")
    st.progress(min(prob, 1.0))

    st.subheader("Why this score? (SHAP explanation)")

    shap_vals = explainer.shap_values(
        input_data, check_additivity=False)

    if isinstance(shap_vals, list):
        sv = shap_vals[1][0]
        ev = explainer.expected_value[1]
    else:
        sv = shap_vals[0]
        ev = explainer.expected_value

    exp = shap.Explanation(
        values=sv,
        base_values=ev,
        data=input_data.iloc[0].values,
        feature_names=feature_cols
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(exp, max_display=10, show=False)
    st.pyplot(fig)
    plt.close()

    st.caption("Red bars push risk higher. Blue bars push risk lower. Bar length = strength of impact.")

else:
    st.info("Adjust the sliders in the sidebar and click 'Calculate risk score'.")