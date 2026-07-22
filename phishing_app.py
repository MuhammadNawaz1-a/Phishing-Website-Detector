import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Set Page Config
st.set_page_config(
    page_title="Phishing Website Security Detector",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Phishing Website Detector")
st.write("Real-time Machine Learning Security Analysis Tool")
st.markdown("---")

# Function to automatically train model if joblib files are missing
@st.cache_resource
def get_model_and_scaler():
    model_path = "best_phishing_model.joblib"
    scaler_path = "scaler.joblib"
    dataset_path = "Phishing_Legitimate_full.csv"

    # Option A: If joblib files exist, load them directly
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler, "Loaded pre-trained model successfully!"

    # Option B: If CSV exists, train automatically on the fly
    elif os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        
        if 'id' in df.columns:
            df.drop('id', axis=1, inplace=True)
        df.drop_duplicates(inplace=True)

        X = df.drop("CLASS_LABEL", axis=1)
        y = df["CLASS_LABEL"]

        # Proper Split Execution Sequence
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Save for future fast loading
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        return model, scaler, "Model trained automatically from CSV dataset!"

    else:
        return None, None, "Dataset CSV ya Joblib files nahi mili!"

# Load Model & Scaler
model, scaler, status_message = get_model_and_scaler()

if model is None:
    st.error("❌ **Error:** Folder mein `Phishing_Legitimate_full.csv` file rakhein taakay model train ho sakay.")
    st.stop()
else:
    st.sidebar.success(f"✅ {status_message}")

# Sidebar Feature Inputs
st.sidebar.header("⚙️ Feature Inputs")

num_dots = st.sidebar.number_input("NumDots", min_value=0, max_value=20, value=2)
subdomain_level = st.sidebar.number_input("SubdomainLevel", min_value=0, max_value=15, value=1)
path_level = st.sidebar.number_input("PathLevel", min_value=0, max_value=20, value=2)
url_length = st.sidebar.slider("UrlLength", min_value=10, max_value=300, value=50)
num_dash = st.sidebar.slider("NumDash", min_value=0, max_value=30, value=1)
hostname_length = st.sidebar.slider("HostnameLength", min_value=1, max_value=150, value=20)
path_length = st.sidebar.slider("PathLength", min_value=0, max_value=200, value=15)
pct_ext_hyperlinks = st.sidebar.slider("PctExtHyperlinks", min_value=0.0, max_value=1.0, value=0.1)
pct_ext_resource_urls = st.sidebar.slider("PctExtResourceUrls", min_value=0.0, max_value=1.0, value=0.1)

# Categorical/Binary Form Inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌐 Domain Parameters")
    ip_address = st.selectbox("Contains Raw IP Address?", [0, 1])
    no_https = st.selectbox("Missing HTTPS Protocol?", [0, 1])
    at_symbol = st.selectbox("Contains '@' Symbol?", [0, 1])

with col2:
    st.subheader("🔒 Security Indicators")
    insecure_forms = st.selectbox("Insecure Submission Forms?", [0, 1])
    frequent_domain_mismatch = st.selectbox("Domain Mismatch Detected?", [0, 1])
    num_sensitive_words = st.number_input("NumSensitiveWords", min_value=0, max_value=10, value=0)

# Build Complete Feature Input (48 Features baseline mapping)
raw_feature_dict = {
    'NumDots': num_dots,
    'SubdomainLevel': subdomain_level,
    'PathLevel': path_level,
    'UrlLength': url_length,
    'NumDash': num_dash,
    'NumDashInHostname': 0,
    'AtSymbol': at_symbol,
    'TildeSymbol': 0,
    'NumUnderscore': 0,
    'NumPercent': 0,
    'NumQueryComponents': 0,
    'NumAmpersand': 0,
    'NumHash': 0,
    'NumNumericChars': 0,
    'NoHttps': no_https,
    'RandomString': 0,
    'IpAddress': ip_address,
    'DomainInSubdomains': 0,
    'DomainInPaths': 0,
    'HttpsInHostname': 0,
    'HostnameLength': hostname_length,
    'PathLength': path_length,
    'QueryLength': 0,
    'DoubleSlashInPath': 0,
    'NumSensitiveWords': num_sensitive_words,
    'EmbeddedBrandName': 0,
    'PctExtHyperlinks': pct_ext_hyperlinks,
    'PctExtResourceUrls': pct_ext_resource_urls,
    'ExtFavicon': 0,
    'InsecureForms': insecure_forms,
    'RelativeFormAction': 0,
    'ExtFormAction': 0,
    'AbnormalFormAction': 0,
    'PctNullSelfRedirectHyperlinks': 0.0,
    'FrequentDomainNameMismatch': frequent_domain_mismatch,
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

# Prediction Section
st.markdown("---")
if st.button("🚀 Predict Security Status"):
    input_df = pd.DataFrame([raw_feature_dict])
    
    # Standardize input using fitted Scaler
    scaled_features = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(scaled_features)[0]
    prediction_proba = model.predict_proba(scaled_features)[0]
    
    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ **PHISHING DETECTED!** (Confidence: {prediction_proba[1]*100:.2f}%)")
    else:
        st.success(f"✅ **LEGITIMATE WEBSITE** (Confidence: {prediction_proba[0]*100:.2f}%)")
