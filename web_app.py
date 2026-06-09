import streamlit as st
import pandas as pd
import pickle

# ==========================================
# 1. PAGE CONFIGURATION & MODEL LOADING
# ==========================================
st.set_page_config(page_title="HR Analytics", page_icon="🏢")

# @st.cache_resource keeps the model loaded in memory so the web app is lightning fast
@st.cache_resource
def load_model():
    with open('rf_model.pkl', 'rb') as f:
        return pickle.load(f)

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

# Convert "Yes/No" to 1/0 for the AI
ot = 1 if ot_text == "Yes" else 0 

# ==========================================
# 3. PREDICTION ENGINE
# ==========================================
# When the user clicks the Analyze button...
if st.button("Analyze Flight Risk", type="primary"):
    
    # Format the data exactly how the Random Forest expects it
    input_data = pd.DataFrame({'Satisfaction': [sat], 'Years': [years], 'Income': [income], 'Overtime': [ot]})
    
    # Get the exact probability percentage
    risk_prob = model.predict_proba(input_data)[0][1] * 100
    
    # Display massive metric on the screen
    st.metric(label="Calculated Flight Risk", value=f"{risk_prob:.1f}%")
    
    # Display color-coded warnings
    if risk_prob > 50:
        st.error("🚨 **HIGH RISK**: This employee is highly likely to quit. HR Intervention recommended.")
    else:
        st.success("✅ **STABLE**: This employee is not currently a flight risk.")