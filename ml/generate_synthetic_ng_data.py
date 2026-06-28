import pandas as pd
import numpy as np
import os

# Set seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(num_samples=10000):
    """
    Generates a synthetic dataset for Nigerian credit scoring,
    incorporating traditional and alternative data features inspired by Zindi challenges.
    """

    # --- 1. Basic Demographics & Traditional Financials ---
    age = np.random.randint(18, 65, num_samples)

    # Annual income (NGN) - roughly 600k to 15m
    # Using log-normal distribution to skew towards lower/middle income
    annual_income = np.random.lognormal(mean=14, sigma=1, size=num_samples)
    annual_income = np.clip(annual_income, 600000, 25000000)
    annual_income = np.round(annual_income / 1000) * 1000 # Round to nearest 1000

    # Loan amount (NGN) - typically a fraction of income
    loan_amount = annual_income * np.random.uniform(0.05, 0.4, num_samples)
    loan_amount = np.round(loan_amount / 1000) * 1000

    # Loan term in months
    loan_term_months = np.random.choice([3, 6, 9, 12, 18, 24], num_samples, p=[0.3, 0.4, 0.15, 0.1, 0.03, 0.02])

    # Loan purpose
    purposes = ['personal', 'business', 'education', 'medical', 'emergency']
    loan_purpose = np.random.choice(purposes, num_samples, p=[0.5, 0.3, 0.1, 0.05, 0.05])

    # Employment Status
    emp_statuses = ['Employed', 'Self-employed', 'Unemployed']
    employment_status = np.random.choice(emp_statuses, num_samples, p=[0.4, 0.5, 0.1])

    # --- 2. Nigerian Context & Alternative Data ---

    # BVN linkage
    has_bvn = np.random.choice([True, False], num_samples, p=[0.9, 0.1])

    # Telco Provider
    telcos = ['MTN', 'Airtel', 'Glo', '9mobile']
    telco_provider = np.random.choice(telcos, num_samples, p=[0.4, 0.3, 0.2, 0.1])

    # Monthly airtime spend (NGN)
    monthly_airtime_spend = np.random.lognormal(mean=7, sigma=1.2, size=num_samples)
    monthly_airtime_spend = np.clip(monthly_airtime_spend, 500, 50000)

    # Active betting account (Alternative risk signal)
    active_betting_account = np.random.choice([True, False], num_samples, p=[0.3, 0.7])

    # USSD Bank Usage (Proxy for digital literacy/activity)
    ussd_bank_usage = np.random.poisson(lam=15, size=num_samples)

    # --- 3. Zindi Challenge Features: New vs Repeat Customer ---
    is_repeat_customer = np.random.choice([True, False], num_samples, p=[0.6, 0.4])

    # For repeat customers, add behavioral history
    previous_loans_count = np.where(is_repeat_customer, np.random.randint(1, 10, num_samples), 0)
    previous_late_payments = np.where(is_repeat_customer, np.random.poisson(lam=1, size=num_samples), 0)
    previous_late_payments = np.clip(previous_late_payments, 0, previous_loans_count)

    # --- 4. Target Generation (Good = 1, Bad = 0) ---
    # We create a hidden "risk score" linearly combined from features + noise

    risk_score_hidden = np.zeros(num_samples)

    # Positive factors (increase likelihood of repayment)
    risk_score_hidden += (annual_income / 5000000) * 0.5
    risk_score_hidden += (age > 25) * 0.2
    risk_score_hidden += (employment_status == 'Employed') * 0.4
    risk_score_hidden += (employment_status == 'Self-employed') * 0.2
    risk_score_hidden += has_bvn * 1.0
    risk_score_hidden += (monthly_airtime_spend / 10000) * 0.3
    risk_score_hidden += (ussd_bank_usage / 20) * 0.2

    # Repeat customer specific positive factors
    risk_score_hidden += is_repeat_customer * 0.5
    risk_score_hidden += np.where(previous_loans_count > 0, 0.3 * (1 - (previous_late_payments / np.maximum(previous_loans_count, 1))), 0)

    # Negative factors (decrease likelihood of repayment / increase risk)
    risk_score_hidden -= (loan_amount / annual_income) * 1.5 # Debt to income proxy
    risk_score_hidden -= active_betting_account * 0.6
    risk_score_hidden -= (employment_status == 'Unemployed') * 0.8
    risk_score_hidden -= (previous_late_payments > 2) * 1.0

    # Add random noise for realistic model fitting challenge
    noise = np.random.normal(0, 0.8, num_samples)
    risk_score_hidden += noise

    # Convert hidden continuous score to probability using sigmoid function
    probability_good = 1 / (1 + np.exp(-risk_score_hidden))

    # Threshold probability to create binary class
    # To mimic real-world Nigerian default rates (e.g. 15-25% default rate)
    # We want ~80% Good (1) and ~20% Bad (0)
    threshold = np.percentile(probability_good, 20)
    loan_status = (probability_good > threshold).astype(int)

    # Calculate mock Credit Risk Score based on the probability (300 to 850)
    # This aligns with the Zindi challenge requirement for a scoring function
    credit_risk_score = 300 + (probability_good * 550)
    credit_risk_score = np.round(credit_risk_score).astype(int)

    # Determine Risk Band
    conditions = [
        (credit_risk_score >= 700),
        (credit_risk_score >= 550) & (credit_risk_score < 700),
        (credit_risk_score < 550)
    ]
    choices = ['Low', 'Medium', 'High']
    risk_band = np.select(conditions, choices, default='Unknown')

    # Assemble dataframe
    data = pd.DataFrame({
        'age': age,
        'annual_income': annual_income,
        'loan_amount': loan_amount,
        'loan_term_months': loan_term_months,
        'loan_purpose': loan_purpose,
        'employment_status': employment_status,
        'has_bvn': has_bvn,
        'telco_provider': telco_provider,
        'monthly_airtime_spend': monthly_airtime_spend,
        'active_betting_account': active_betting_account,
        'ussd_bank_usage': ussd_bank_usage,
        'is_repeat_customer': is_repeat_customer,
        'previous_loans_count': previous_loans_count,
        'previous_late_payments': previous_late_payments,
        # Target variable
        'loan_status': loan_status,
        # Mock API Outputs for reference/evaluation
        'mock_probability': probability_good,
        'mock_credit_score': credit_risk_score,
        'mock_risk_band': risk_band
    })

    return data

if __name__ == "__main__":
    print("Generating Nigerian synthetic credit dataset...")
    df = generate_synthetic_data(num_samples=25000)

    output_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'synthetic_ng_credit_data.csv')
    df.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")
    print(f"Total records: {len(df)}")
    print(f"Default rate (Bad = 0): {100 - (df['loan_status'].mean() * 100):.2f}%")
    print(f"Features: {list(df.columns)}")
    print("\nSample Data:")
    print(df.head())
