import streamlit as st
import numpy as np
import pandas as pd
import re
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Page Config
st.set_page_config(
    page_title="Phishing Website Security Detector",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Phishing Website Detector")
st.write("Paste any URL below to analyze if it's Safe or Phishing in real-time.")
st.markdown("---")

# Function to train model dynamically if pre-baked models aren't loaded
@st.cache_resource
def train_model_from_df(df):
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    df = df.drop_duplicates()

    X = df.drop("CLASS_LABEL", axis=1)
    y = df["CLASS_LABEL"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler

# Feature Extractor Function (URL string ko 48 features mein convert karne ke liye)
def extract_features_from_url(url):
    # Ensure protocol exists for parsing
    if not url.startswith(('http://', 'https://')):
        parsed_url = urlparse('http://' + url)
        has_https = 0
    else:
        parsed_url = urlparse(url)
        has_https = 1 if url.startswith('https://') else 0

    hostname = parsed_url.netloc or parsed_url.path.split('/')[0]
    path = parsed_url.path

    # Extract Structural URL Features
    num_dots = url.count('.')
    subdomain_level = max(0, len(hostname.split('.')) - 2)
    path_level = len([p for p in path.split('/') if p])
    url_length = len(url)
    num_dash = url.count('-')
    num_dash_hostname = hostname.count('-')
    at_symbol = 1 if '@' in url else 0
    tilde_symbol = 1 if '~' in url else 0
    num_underscore = url.count('_')
    num_percent = url.count('%')
    num_query_comp = len(parsed_url.query.split('&')) if parsed_url.query else 0
    num_ampersand = url.count('&')
    num_hash = url.count('#')
    num_numeric = sum(c.isdigit() for c in url)
    no_https = 1 if has_https == 0 else 0
    
    # Check IP address usage
    ip_pattern = r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"
    ip_address = 1 if re.match(ip_pattern, hostname) else 0

    hostname_length = len(hostname)
    path_length = len(path)
    query_length = len(parsed_url.query)
    
    # Sensitive words check
    sensitive_words = ['login', 'bank', 'secure', 'update', 'verify', 'account', 'signin', 'wp', 'cmd']
    num_sensitive_words = sum(1 for word in sensitive_words if word in url.lower())

    # Build exact matching feature dictionary for the 48 features
    features = {
        'NumDots': num_dots,
        'SubdomainLevel': subdomain_level,
        'PathLevel': path_level,
        'UrlLength': url_length,
        'NumDash': num_dash,
        'NumDashInHostname': num_dash_hostname,
        'AtSymbol': at_symbol,
        'TildeSymbol': tilde_symbol,
        'NumUnderscore': num_underscore,
        'NumPercent': num_percent,
        'NumQueryComponents': num_query_comp,
        'NumAmpersand': num_ampersand,
        'NumHash': num_hash,
        'NumNumericChars': num_numeric,
        'NoHttps': no_https,
        'RandomString': 0,
        'IpAddress': ip_address,
        'DomainInSubdomains': 0,
        'DomainInPaths': 0,
        'HttpsInHostname': 0,
        'HostnameLength': hostname_length,
        'PathLength': path_length,
        'QueryLength': query_length,
        'DoubleSlashInPath': 1 if '//' in path else 0,
        'NumSensitiveWords': num_sensitive_words,
        'EmbeddedBrandName': 0,
        'PctExtHyperlinks': 0.0,
        'PctExtResourceUrls': 0.0,
        'ExtFavicon': 0,
        'InsecureForms': 0,
        'RelativeFormAction': 0,
        'ExtFormAction': 0,
        'AbnormalFormAction': 0,
        'PctNullSelfRedirectHyperlinks': 0.0,
        'FrequentDomainNameMismatch': 0,
        'FakeLinkInStatusBar': 0,
        'RightClickDisabled': 0,
        'PopUpWindow': 0,
        'SubmitInfoToEmail': 0,
        'IframeOrFrame': 0,
        'MissingTitle': 0,
        'ImagesOnlyInForm': 0,
        'SubdomainLevelRT': 0,
        'UrlLengthRT': 0,
        'PctExtResourceUrlsRT': 0,
        'AbnormalExtFormActionR': 0,
        'ExtMetaScriptLinkRT': 0,
        'PctExtNullSelfRedirectHyperlinksRT': 0
    }
    
    return pd.DataFrame([features])

# Handle Model Loading / Dataset Fallback
model = None
scaler = None

if os.path.exists("best_phishing_model.joblib") and os.path.exists("scaler.joblib"):
    model = joblib.load("best_phishing_model.joblib")
    scaler = joblib.load("scaler.joblib")

elif os.path.exists("Phishing_Legitimate_full.csv"):
    df = pd.read_csv("Phishing_Legitimate_full.csv")
    model, scaler = train_model_from_df(df)

else:
    st.sidebar.warning("⚠️ Local model/dataset not found.")
    uploaded_file = st.sidebar.file_uploader("Upload `Phishing_Legitimate_full.csv`", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        model, scaler = train_model_from_df(df)

if model is None or scaler is None:
    st.info("👈 Please upload your `Phishing_Legitimate_full.csv` dataset in the sidebar once to initialize the ML model.")
    st.stop()

# MAIN INTERFACE: Simple URL Input Box
target_url = st.text_input("🔗 Enter Website URL:", placeholder="e.g., https://example.com or http://login-paypal-verify.com")

if st.button("🔍 Check URL Security", type="primary"):
    if not target_url.strip():
        st.warning("Please enter a valid URL first!")
    else:
        try:
            # 1. Feature Extraction from raw URL string
            input_features_df = extract_features_from_url(target_url)
            
            # 2. Scale features using fitted scaler
            scaled_features = scaler.transform(input_features_df)
            
            # 3. Model Prediction
            prediction = model.predict(scaled_features)[0]
            prediction_proba = model.predict_proba(scaled_features)[0]
            
            st.markdown("---")
            st.subheader("📊 Analysis Result")
            
            if prediction == 1:
                st.error(f"⚠️ **PHISHING / SPAM URL DETECTED!**")
                st.write(f"Phishing Probability: **{prediction_proba[1]*100:.2f}%**")
                st.write("❌ **Recommendation:** Do NOT open or enter personal credentials on this link.")
            else:
                st.success(f"✅ **LEGITIMATE / SAFE WEBSITE**")
                st.write(f"Safe Confidence Score: **{prediction_proba[0]*100:.2f}%**")
                st.write("✔️ **Status:** This URL structure looks clean and normal.")
                
        except Exception as e:
            st.error(f"Error analyzing URL: {str(e)}")
