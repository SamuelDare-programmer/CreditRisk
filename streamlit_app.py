import streamlit as st
import requests
import os
import datetime
import base64
import pandas as pd

# ==========================================
# Page Configuration & CSS
# ==========================================
st.set_page_config(
    page_title="Credit Risk Scoring System",
    page_icon="🏦",
    layout="wide"
)

# Custom premium CSS
st.markdown("""
<style>
    /* Base theme */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers and typography */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600;
        color: #ffffff;
    }
    
    /* Metric cards (glassmorphism) */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-title {
        font-size: 1.1rem;
        color: #a0a0a0;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    /* Status Colors */
    .status-low { color: #00d4aa; }
    .status-medium { color: #ffb347; }
    .status-high { color: #ff4757; }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 10px;
    }
    .badge-low {
        background-color: rgba(0, 212, 170, 0.1);
        color: #00d4aa;
        border: 1px solid rgba(0, 212, 170, 0.3);
    }
    .badge-medium {
        background-color: rgba(255, 179, 71, 0.1);
        color: #ffb347;
        border: 1px solid rgba(255, 179, 71, 0.3);
    }
    .badge-high {
        background-color: rgba(255, 71, 87, 0.1);
        color: #ff4757;
        border: 1px solid rgba(255, 71, 87, 0.3);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.8em;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Header Area
# ==========================================
st.markdown("<h1>🏦 Credit Risk Scoring System</h1>", unsafe_allow_html=True)
st.markdown("<h3>AI-Powered Loan Default Prediction with Explainable SHAP Analysis</h3>", unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# Sidebar: Input Form
# ==========================================
with st.sidebar:
    st.header("Borrower Profile")
    with st.form("borrower_form"):
        # Section 1: Demographics
        st.subheader("👤 Demographics")
        birthdate = st.date_input("Date of Birth", value=datetime.date(1990, 1, 1))
        employment_status_clients = st.selectbox("Employment Status", options=["permanent", "self-employed", "contract", "unemployed"])
        bank_name_clients = st.selectbox("Bank Name", options=["gt bank", "first bank", "access bank", "zenith bank", "uba", "other"])
        bank_account_type = st.selectbox("Bank Account Type", options=["savings", "current"])
        
        col1, col2 = st.columns(2)
        with col1:
            longitude_gps = st.number_input("Longitude", value=3.3792, format="%.4f")
        with col2:
            latitude_gps = st.number_input("Latitude", value=6.5244, format="%.4f")

        # Section 2: Loan Details
        st.subheader("💰 Loan Details")
        loanamount = st.number_input("Loan Amount (NGN)", min_value=1000.0, value=50000.0, step=1000.0, format="%.0f")
        totaldue = st.number_input("Total Due (NGN)", min_value=1000.0, value=55000.0, step=1000.0, format="%.0f")
        termdays = st.number_input("Term (Days)", min_value=1, value=30, step=1)
        
        c1, c2 = st.columns(2)
        with c1:
            approveddate = st.date_input("Approved Date", value=datetime.date.today())
        with c2:
            creationdate = st.date_input("Creation Date", value=datetime.date.today())

        # Section 3: Credit History
        st.subheader("📊 Credit History")
        prev_loan_count = st.number_input("Previous Loan Count", min_value=0, value=0)
        prev_total_borrowed = st.number_input("Total Borrowed (NGN)", min_value=0.0, value=0.0, format="%.2f")
        prev_avg_loan_amount = st.number_input("Avg Loan Amount (NGN)", min_value=0.0, value=0.0, format="%.2f")
        prev_max_loan_amount = st.number_input("Max Loan Amount (NGN)", min_value=0.0, value=0.0, format="%.2f")
        prev_avg_totaldue = st.number_input("Avg Total Due (NGN)", min_value=0.0, value=0.0, format="%.2f")
        prev_avg_term_days = st.number_input("Avg Term (Days)", min_value=0.0, value=0.0, format="%.2f")
        prev_late_repayment_count = st.number_input("Late Repayment Count", min_value=0, value=0)
        prev_late_repayment_ratio = st.slider("Late Repayment Ratio", 0.0, 1.0, 0.0, step=0.01)
        prev_repayment_speed_avg = st.number_input("Repayment Speed Avg", value=0.0)

        # Section 4: Alternative Data
        with st.expander("🇳🇬 Nigerian Alternative Data"):
            has_bvn = st.checkbox("BVN Verified", value=True)
            ussd_bank_usage = st.number_input("USSD Bank Usage (last 30d)", min_value=0, value=15)
            monthly_airtime_spend = st.number_input("Monthly Airtime Spend", min_value=0.0, value=5000.0)
            active_betting_account = st.checkbox("Active Betting Account", value=False)
            telco_provider = st.selectbox("Telco Provider", options=["MTN", "Airtel", "Glo", "9mobile"])
            annual_income = st.number_input("Annual Income", min_value=0.0, value=2000000.0)

        # Submit Button
        submitted = st.form_submit_button("🔍 Analyze Risk", use_container_width=True)

# ==========================================
# Main Area: Results Display
# ==========================================
if submitted:
    with st.spinner("Analyzing borrower risk profile..."):
        # Build payload
        payload = {
            "birthdate": birthdate.strftime("%Y-%m-%d"),
            "bank_account_type": bank_account_type,
            "bank_name_clients": bank_name_clients,
            "longitude_gps": longitude_gps,
            "latitude_gps": latitude_gps,
            "employment_status_clients": employment_status_clients,
            
            "loanamount": loanamount,
            "totaldue": totaldue,
            "termdays": termdays,
            
            "approveddate": approveddate.strftime("%Y-%m-%d"),
            "creationdate": creationdate.strftime("%Y-%m-%d"),
            
            "prev_loan_count": prev_loan_count,
            "prev_total_borrowed": prev_total_borrowed,
            "prev_avg_loan_amount": prev_avg_loan_amount,
            "prev_max_loan_amount": prev_max_loan_amount,
            "prev_avg_totaldue": prev_avg_totaldue,
            "prev_avg_term_days": prev_avg_term_days,
            "prev_late_repayment_count": prev_late_repayment_count,
            "prev_late_repayment_ratio": prev_late_repayment_ratio,
            "prev_repayment_speed_avg": prev_repayment_speed_avg,
            
            "has_bvn": has_bvn,
            "ussd_bank_usage": ussd_bank_usage,
            "monthly_airtime_spend": monthly_airtime_spend,
            "active_betting_account": active_betting_account,
            "telco_provider": telco_provider,
            "annual_income": annual_income,
        }

        API_URL = os.getenv("API_URL", "http://localhost:8000")
        API_KEY = os.getenv("API_KEY", "dev-secret-key-change-in-prod")
        
        try:
            response = requests.post(
                f"{API_URL}/api/v1/score",
                json=payload,
                headers={"X-API-Key": API_KEY},
                timeout=30,
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Determine styling classes based on risk category
                cat = data.get("risk_category", "").lower()
                if "low" in cat:
                    score_class = "status-low"
                    badge_class = "badge-low"
                    rec_class = "status-low"
                    rec_icon = "✅"
                elif "medium" in cat:
                    score_class = "status-medium"
                    badge_class = "badge-medium"
                    rec_class = "status-medium"
                    rec_icon = "⚠️"
                else:
                    score_class = "status-high"
                    badge_class = "badge-high"
                    rec_class = "status-high"
                    rec_icon = "❌"
                    
                risk_score_pct = data.get("risk_score", 0) * 100
                category_text = data.get("risk_category", "Unknown").replace("_", " ").upper()
                recommendation_text = data.get("recommendation", "Unknown").upper()

                # Row 1: Metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Risk Score</div>
                        <div class="metric-value {score_class}">{risk_score_pct:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Risk Category</div>
                        <div class="badge {badge_class}">{category_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Recommendation</div>
                        <div class="metric-value {rec_class}">{rec_icon} {recommendation_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)

                # Row 2: SHAP Plot
                if data.get("shap_waterfall_plot"):
                    st.subheader("📊 SHAP Feature Contribution Analysis")
                    st.caption("This waterfall plot shows how each feature pushes the prediction from the base value to the final risk score.")
                    img_bytes = base64.b64decode(data["shap_waterfall_plot"])
                    st.image(img_bytes, use_container_width=True)
                    st.markdown("<br>", unsafe_allow_html=True)

                # Row 3: Risk Factors
                if data.get("risk_factors"):
                    st.subheader("🔍 Top Risk Factors")
                    factors_df = pd.DataFrame(data["risk_factors"])
                    
                    # Add emoji direction indicators
                    factors_df["direction"] = factors_df["direction"].map({
                        "increased_risk": "⬆️ Increased Risk",
                        "decreased_risk": "⬇️ Decreased Risk",
                    }).fillna(factors_df["direction"])
                    
                    factors_df["impact"] = factors_df["impact"].round(4)
                    factors_df.columns = ["Feature", "Impact", "Direction"]
                    st.dataframe(factors_df, use_container_width=True, hide_index=True)

            else:
                st.error(f"❌ API Error ({response.status_code}): {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to the API. Make sure the FastAPI backend is running on port 8000.")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}")

# ==========================================
# Footer
# ==========================================
st.markdown("---")
st.markdown(
    "<div class='footer'>"
    "Credit Risk Scoring System | Department of Computer Science, University of Benin (UNIBEN)"
    "<br>Final Year Project — Madehin Oluwadamilare Samuel"
    "</div>",
    unsafe_allow_html=True,
)
