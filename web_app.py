import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. PAGE CONFIGURATION & CLOUD TRAINING
# ==========================================
st.set_page_config(page_title="HR Analytics", page_icon="🏢")

# @st.cache_resource ensures the AI only trains once when the server boots
@st.cache_resource
def load_model():
    # Bypass the .pkl file completely and train the AI right here in the cloud!
    np.random.seed(42)
    n_samples = 5000
    
    satisfaction = np.random.randint(1, 11, n_samples)
    years = np.random.randint(0, 20, n_samples)
    income = np.random.randint(3000, 15000, n_samples)
    overtime = np.random.randint(0, 2, n_samples)
    
    prob = 0.5 - (satisfaction * 0.04) + (overtime * 0.2) - (income * 0.00001) + (np.random.randn(n_samples) * 0.1)
    attrition = (prob > 0.45).astype(int)
    
    X = pd.DataFrame({'Satisfaction': satisfaction, 'Years': years, 'Income': income, 'Overtime': overtime})
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, attrition)
    
    return model

model = load_model()

# ==========================================
# 2. WEB APP USER INTERFACE
# ==========================================
st.title("🏃‍♂️ Employee Flight-Risk Dashboard")
st.markdown("Adjust the employee metrics in the sidebar to predict the probability of them leaving the company using our **Random Forest AI**.")

# -- Sidebar Controls --
st.sidebar.header("Employee Stats")
sat = st.sidebar.slider("Job Satisfaction (1-10)", min_value=1, max_value=10, value=5)
years = st.sidebar.slider("Years at Company", min_value=0, max_value=20, value=3)
income = st.sidebar.slider("Monthly Income ($)", min_value=3000, max_value=15000, value=5000, step=500)
ot_text = st.sidebar.selectbox("Works Overtime?", ["Yes", "No"])

ot = 1 if ot_text == "Yes" else 0 

# ==========================================
# 3. PREDICTION ENGINE
# ==========================================
if st.button("Analyze Flight Risk", type="primary"):
    
    input_data = pd.DataFrame({'Satisfaction': [sat], 'Years': [years], 'Income': [income], 'Overtime': [ot]})
    risk_prob = model.predict_proba(input_data)[0][1] * 100
    
    st.metric(label="Calculated Flight Risk", value=f"{risk_prob:.1f}%")
    
    if risk_prob > 50:
        st.error("🚨 **HIGH RISK**: This employee is highly likely to quit. HR Intervention recommended.")
    else:
        st.success("✅ **STABLE**: This employee is not currently a flight risk.")