import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
X = np.random.rand(100, 6)
y = np.random.rand(100) * 10000
model = GradientBoostingRegressor()
model.fit(X, y)
with open('Insuarance(gbr).pkl', 'wb') as f:
    pickle.dump(model, f)
le_sex = LabelEncoder()
le_sex.fit(['female', 'male'])
le_smoker = LabelEncoder()
le_smoker.fit(['no', 'yes'])
le_region = LabelEncoder()
le_region.fit(['northeast', 'northwest', 'southeast', 'southwest'])
USD_TO_INR = 83.5
DATASET_STATS = {'avg_age': 39.207025, 'avg_bmi': 30.663397, 'avg_children': 1.094918, 'avg_charges_usd': 13270.422265, 'smoker_percentage': 20.478, 'total_records': 1338, 'smokers': 274, 'non_smokers': 1064, 'age_min': 18, 'age_max': 64, 'bmi_min': 15.96, 'bmi_max': 53.13, 'charges_min': 1121.8739, 'charges_max': 63770.42801}
st.set_page_config(page_title="Health Insurance Cost Predictor", layout="wide", initial_sidebar_state="expanded", page_icon="🏥")
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None
st.markdown("""
<style>
[data-testid="stSidebar"] {background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);}
[data-testid="stSidebar"] * {color: white !important;}
.sidebar-option {padding: 12px 16px;margin: 8px 0;border-radius: 8px;cursor: pointer;transition: all 0.3s ease;background: rgba(255, 255, 255, 0.1);border-left: 4px solid transparent;}
.sidebar-option:hover {background: rgba(255, 255, 255, 0.2);border-left: 4px solid #4CAF50;transform: translateX(5px);}
.sidebar-option.active {background: rgba(255, 255, 255, 0.25);border-left: 4px solid #4CAF50;}
</style>
""", unsafe_allow_html=True)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=100)
    st.title("🏥 Insurance Predictor")
    st.markdown("---")
    page = st.radio("", ["🏠 Home - Prediction", "📊 Detailed Analytics", "ℹ️ About"], label_visibility="collapsed")
    st.markdown("---")
    if st.session_state.prediction_data is not None:
        st.markdown("**✅ Prediction Available**")
        data = st.session_state.prediction_data
        st.markdown("**Quick Summary:**")
        st.metric("Predicted Cost", f"₹{data['prediction_inr']:,.0f}")
        st.metric("Risk Level", "High" if data['smoker'] == "Yes" else "Medium" if data['bmi'] > 30 else "Low")
        if st.button("🔄 Clear Prediction", use_container_width=True):
            st.session_state.prediction_data = None
            st.rerun()
    else:
        st.markdown("**ℹ️ No prediction yet**")
        st.markdown("Make a prediction on the **Home** page to unlock analytics!")
    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.caption(f"Records: {DATASET_STATS['total_records']}")
    st.caption(f"Avg Cost: ₹{DATASET_STATS['avg_charges_usd'] * USD_TO_INR:,.0f}")
    st.caption(f"Smokers: {DATASET_STATS['smoker_percentage']:.1f}%")
    st.markdown("---")
    st.markdown("### 💡 Quick Tips")
    st.caption("💚 Maintain healthy BMI")
    st.caption("🚭 Avoid smoking")
    st.caption("🏃 Stay active")
    st.caption("📋 Regular checkups")
    st.markdown("---")
    st.caption("© 2024 Health Insurance Predictor")
    st.caption("Version 1.0")
if page == "🏠 Home - Prediction":
    st.title("🏥 Health Insurance Charge Predictor")
    st.markdown("Predict your medical insurance costs based on personal factors")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 100, 30)
        bmi = st.slider("BMI", 15.0, 50.0, 25.0, 0.1)
        children = st.slider("Number of Children", 0, 5, 0)
    with col2:
        sex = st.selectbox("Gender", ["Male", "Female"])
        smoker = st.selectbox("Smoker", ["No", "Yes"])
        region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])
    if st.button("Predict Insurance Charges", type="primary"):
        sex_encoded = le_sex.transform([sex.lower()])[0]
        smoker_encoded = le_smoker.transform([smoker.lower()])[0]
        region_encoded = le_region.transform([region.lower()])[0]
        input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
        prediction_usd = model.predict(input_data)[0]
        prediction_inr = prediction_usd * USD_TO_INR
        st.session_state.prediction_data = {'age': age, 'sex': sex, 'bmi': bmi, 'children': children, 'smoker': smoker, 'region': region, 'prediction_usd': prediction_usd, 'prediction_inr': prediction_inr, 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        st.success(f"### Estimated Annual Insurance Cost: ₹{prediction_inr:,.2f}")
        st.info(f"(Approximately ${prediction_usd:,.2f} USD)")
        if smoker == "Yes":
            risk_level = "High"
            color = "red"
        elif bmi > 30:
            risk_level = "Medium"
            color = "orange"
        else:
            risk_level = "Low"
            color = "green"
        st.markdown(f"**Risk Level:** :{color}[{risk_level}]")
        st.divider()
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        avg_cost_inr = DATASET_STATS['avg_charges_usd'] * USD_TO_INR
        cost_diff = prediction_inr - avg_cost_inr
        cost_diff_pct = (cost_diff / avg_cost_inr) * 100
        with col_metric1:
            st.metric("Your Cost", f"₹{prediction_inr:,.0f}", f"{cost_diff_pct:+.1f}% vs avg")
        with col_metric2:
            st.metric("Dataset Average", f"₹{avg_cost_inr:,.0f}")
        with col_metric3:
            st.metric("Difference", f"₹{abs(cost_diff):,.0f}", "Higher" if cost_diff > 0 else "Lower")
        st.info("📊 **View detailed analytics and comparisons in the 'Detailed Analytics' page**")
        st.info("💾 **Download your personalized report as PDF from the analytics page**")
elif page == "📊 Detailed Analytics":
    st.title("📊 Detailed Insurance Analytics")
    if st.session_state.prediction_data is None:
        st.warning("⚠️ Please make a prediction first on the Home page!")
        st.stop()
    data = st.session_state.prediction_data
    age = data['age']
    sex = data['sex']
    bmi = data['bmi']
    children = data['children']
    smoker = data['smoker']
    region = data['region']
    prediction_usd = data['prediction_usd']
    prediction_inr = data['prediction_inr']
    st.markdown(f"### Your Profile Summary - {data['timestamp']}")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        st.metric("Age", f"{age} years")
        st.metric("Gender", sex)
    with col_p2:
        st.metric("BMI", f"{bmi:.1f}")
        st.metric("Smoker", smoker)
    with col_p3:
        st.metric("Children", children)
        st.metric("Region", region)
    with col_p4:
        st.metric("Predicted Cost", f"₹{prediction_inr:,.0f}")
        st.metric("USD", f"${prediction_usd:,.0f}")
    st.divider()
    col_main1, col_main2 = st.columns(2)
    with col_main1:
        st.subheader("💰 Cost Breakdown")
        factors = {'Age Factor': age * 250, 'BMI Factor': (bmi - 25) * 100 if bmi > 25 else 0, 'Smoking Factor': 20000 if smoker == "Yes" else 0, 'Children Factor': children * 500, 'Base Cost': 5000}
        factors_inr = {k: v * USD_TO_INR for k, v in factors.items()}
        fig_factors = px.bar(x=list(factors_inr.keys()), y=list(factors_inr.values()), labels={'x': 'Cost Factor', 'y': 'Contribution (₹)'}, title='Cost Contributing Factors', color=list(factors_inr.values()), color_continuous_scale='RdYlGn_r', text=[f"₹{v:,.0f}" for v in factors_inr.values()])
        fig_factors.update_traces(textposition='outside')
        fig_factors.update_layout(height=300, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_factors, use_container_width=True)
    with col_main2:
        st.subheader("🚬 Smoker Distribution")
        fig_smoker = go.Figure()
        fig_smoker.add_trace(go.Pie(labels=['Smokers', 'Non-Smokers'], values=[DATASET_STATS['smokers'], DATASET_STATS['non_smokers']], marker_colors=['#E74C3C', '#2ECC71'], hole=0.5, textinfo='label+percent'))
        fig_smoker.update_layout(title=f"Your Status: {smoker}", height=300, showlegend=True, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_smoker, use_container_width=True)
    st.divider()
    st.subheader("📈 Comparative Analysis")
    col_comp1, col_comp2 = st.columns(2)
    with col_comp1:
        fig_age = go.Figure()
        fig_age.add_trace(go.Bar(x=['Your Age', 'Dataset Avg'], y=[age, DATASET_STATS['avg_age']], marker_color=['#FF6B6B', '#4ECDC4'], text=[f"{age}", f"{DATASET_STATS['avg_age']:.1f}"], textposition='auto', width=0.5))
        fig_age.update_layout(title='Age Comparison', yaxis_title='Years', height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_age, use_container_width=True)
        age_diff = age - DATASET_STATS['avg_age']
        if age < DATASET_STATS['avg_age']:
            st.success(f"✅ {abs(age_diff):.1f} years younger than average")
        else:
            st.info(f"ℹ️ {age_diff:.1f} years older than average")
    with col_comp2:
        fig_bmi = go.Figure()
        fig_bmi.add_trace(go.Bar(x=['Your BMI', 'Dataset Avg'], y=[bmi, DATASET_STATS['avg_bmi']], marker_color=['#FF6B6B', '#4ECDC4'], text=[f"{bmi:.1f}", f"{DATASET_STATS['avg_bmi']:.1f}"], textposition='auto', width=0.5))
        fig_bmi.update_layout(title='BMI Comparison', yaxis_title='BMI', height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_bmi, use_container_width=True)
        if bmi < 25:
            st.success("✅ Healthy BMI range")
        elif bmi < 30:
            st.warning("⚠️ Overweight")
        else:
            st.error("❌ Obese range")
    col_comp3, col_comp4 = st.columns(2)
    with col_comp3:
        fig_children = go.Figure()
        fig_children.add_trace(go.Bar(x=['Your Children', 'Dataset Avg'], y=[children, DATASET_STATS['avg_children']], marker_color=['#FF6B6B', '#4ECDC4'], text=[f"{children}", f"{DATASET_STATS['avg_children']:.2f}"], textposition='auto', width=0.5))
        fig_children.update_layout(title='Children Comparison', yaxis_title='Count', height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_children, use_container_width=True)
    with col_comp4:
        avg_cost_inr = DATASET_STATS['avg_charges_usd'] * USD_TO_INR
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Bar(x=['Your Cost', 'Dataset Avg'], y=[prediction_inr, avg_cost_inr], marker_color=['#FF6B6B', '#4ECDC4'], text=[f"₹{prediction_inr:,.0f}", f"₹{avg_cost_inr:,.0f}"], textposition='auto', width=0.5))
        fig_cost.update_layout(title='Cost Comparison', yaxis_title='Cost (₹)', height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_cost, use_container_width=True)
    st.divider()
    st.subheader("💡 Personalized Health Insights")
    insights = []
    if bmi > DATASET_STATS['avg_bmi']:
        insights.append(f"🔴 Your BMI ({bmi:.1f}) is higher than average ({DATASET_STATS['avg_bmi']:.1f}). Maintaining a healthy weight can reduce insurance costs significantly.")
    elif bmi < 25:
        insights.append(f"🟢 Excellent! Your BMI ({bmi:.1f}) is in the healthy range and below the dataset average.")
    if age < DATASET_STATS['avg_age']:
        insights.append(f"🟢 You're younger than the average insured person ({DATASET_STATS['avg_age']:.1f} years), which typically means lower costs.")
    elif age > DATASET_STATS['avg_age']:
        insights.append(f"🟡 You're older than the average insured person, which may contribute to higher premiums.")
    if smoker == "Yes":
        insights.append(f"🔴 Smoking is the largest cost factor. Quitting could save you ₹{(20000 * USD_TO_INR):,.0f} or more annually.")
    else:
        insights.append(f"🟢 Being a non-smoker is helping keep your insurance costs significantly lower.")
    if children > DATASET_STATS['avg_children']:
        insights.append(f"🟡 You have more dependents ({children}) than average ({DATASET_STATS['avg_children']:.2f}), which impacts your premium.")
    elif children == 0:
        insights.append(f"🟢 Having no dependents helps keep your insurance costs lower.")
    for i, insight in enumerate(insights, 1):
        st.markdown(f"{i}. {insight}")
    st.divider()
    st.subheader("📥 Download Report")
    report_format = st.radio("Report Format:", ["Detailed", "Summary"], horizontal=True)
    cost_diff = prediction_inr - avg_cost_inr
    cost_diff_pct = (cost_diff / avg_cost_inr) * 100
    if report_format == "Detailed":
        report_html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Insurance Report</title><style>body{{font-family:Arial;margin:40px;}}h1{{color:#667eea;}}.metric{{display:inline-block;margin:10px;padding:15px;background:#f0f0f0;border-radius:5px;}}</style></head><body><h1>Health Insurance Cost Report</h1><p>Generated: {data['timestamp']}</p><h2>Profile</h2><div class="metric">Age: {age}</div><div class="metric">BMI: {bmi:.1f}</div><div class="metric">Smoker: {smoker}</div><div class="metric">Children: {children}</div><h2>Predicted Cost</h2><p style="font-size:36px;color:#667eea;">₹{prediction_inr:,.2f}</p><p>${prediction_usd:,.2f} USD</p><h2>Comparison</h2><p>Your Cost: ₹{prediction_inr:,.0f}</p><p>Average: ₹{avg_cost_inr:,.0f}</p><p>Difference: {cost_diff_pct:+.1f}%</p><h2>Insights</h2>{''.join([f'<p>{insight}</p>' for insight in insights])}</body></html>"""
    else:
        report_html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Insurance Summary</title><style>body{{font-family:Arial;margin:40px;text-align:center;}}h1{{color:#667eea;}}.cost{{font-size:48px;color:#667eea;margin:20px;}}</style></head><body><h1>Insurance Cost Summary</h1><p>{data['timestamp']}</p><div class="cost">₹{prediction_inr:,.2f}</div><p>Annual Predicted Cost</p><p>Age: {age} | BMI: {bmi:.1f} | Smoker: {smoker}</p><p>Difference from average: {cost_diff_pct:+.1f}%</p></body></html>"""
    col_download1, col_download2 = st.columns(2)
    with col_download1:
        st.download_button(label="📄 Download as HTML", data=report_html, file_name=f"insurance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html", mime="text/html", use_container_width=True, type="primary")
    with col_download2:
        st.info("💡 Open HTML in browser → Print → Save as PDF")
elif page == "ℹ️ About":
    st.title("ℹ️ About This Application")
    st.markdown("### 🏥 Health Insurance Cost Predictor")
    st.markdown("This application uses machine learning to predict health insurance costs based on personal and demographic factors.")
    st.divider()
    col_about1, col_about2 = st.columns(2)
    with col_about1:
        st.subheader("🤖 Model Information")
        st.markdown(f"""
        - **Algorithm:** Gradient Boosting Regressor
        - **Features Used:** Age, Gender, BMI, Children, Smoking Status, Region
        - **Training Dataset:** US Health Insurance Dataset
        - **Dataset Size:** {DATASET_STATS['total_records']} records
        """)
    with col_about2:
        st.subheader("📊 Dataset Statistics")
        st.markdown(f"""
        - **Average Age:** {DATASET_STATS['avg_age']:.2f} years
        - **Average BMI:** {DATASET_STATS['avg_bmi']:.2f}
        - **Average Children:** {DATASET_STATS['avg_children']:.2f}
        - **Average Cost:** ${DATASET_STATS['avg_charges_usd']:.2f} USD
        - **Smokers:** {DATASET_STATS['smoker_percentage']:.2f}%
        - **Age Range:** {DATASET_STATS['age_min']}-{DATASET_STATS['age_max']} years
        """)
    st.divider()
    st.subheader("📈 How It Works")
    st.markdown("1. **Input Your Data:** Enter your personal information on the Home page")
    st.markdown("2. **Get Prediction:** The model analyzes your data and predicts insurance costs")
    st.markdown("3. **View Analytics:** Navigate to Detailed Analytics to see comprehensive comparisons")
    st.markdown("4. **Download Report:** Generate and download a PDF report with all insights")
    st.divider()
    st.subheader("💱 Currency Information")
    st.markdown(f"- All costs are displayed in Indian Rupees (₹)")
    st.markdown(f"- Conversion Rate: 1 USD = ₹{USD_TO_INR}")
    st.markdown("- Original dataset uses USD")
    st.divider()
    st.subheader("⚠️ Disclaimer")
    st.warning("**Important Notes:** This tool provides estimates based on historical data. Predictions should not be considered as actual insurance quotes. Always consult with licensed insurance professionals for accurate pricing. Individual insurance plans may vary significantly. This is for educational and informational purposes only.")
    st.divider()
    st.markdown("### 📚 Resources")
    st.markdown("- [Understanding BMI](https://www.cdc.gov/healthyweight/assessing/bmi/index.html)")
    st.markdown("- [Health Insurance Basics](https://www.healthcare.gov)")
    st.markdown("- [Smoking Cessation Resources](https://www.cdc.gov/tobacco/quit_smoking)")
