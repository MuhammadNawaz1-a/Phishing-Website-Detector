import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Set Page Config
st.set_page_config(
    page_title="Phishing Website Security Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stAlert {
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# Application Header
st.title("🛡️ Phishing Website Detector")
st.write("Real-time Machine Learning Security Analysis Tool")
st.markdown("---")

# Load Serialized Artifacts safely
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("best_phishing_model.joblib")
        scaler = joblib.load("scaler.joblib")
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_artifacts()

if model is None or scaler is None:
    st.error("❌ Model or Scaler file not found! Please check if `best_phishing_model.joblib` and `scaler.joblib` are present in the directory.")
    st.stop()

# Layout setup: Feature Input Columns
st.sidebar.header("⚙️ Feature Inputs")
st.sidebar.write("Adjust URL structural parameters:")

# Essential Features from training dataset (matching notebook layout)
num_dots = st.sidebar.number_input("NumDots (Count of dots in URL)", min_value=0, max_value=20, value=2)
subdomain_level = st.sidebar.number_input("SubdomainLevel", min_value=0, max_value=15, value=1)
path_level = st.sidebar.number_input("PathLevel", min_value=0, max_value=20, value=2)
url_length = st.sidebar.slider("UrlLength", min_value=10, max_value=300, value=50)
num_dash = st.sidebar.slider("NumDash", min_value=0, max_value=30, value=1)
hostname_length = st.sidebar.slider("HostnameLength", min_value=1, max_value=150, value=20)
path_length = st.sidebar.slider("PathLength", min_value=0, max_value=200, value=15)
pct_ext_hyperlinks = st.sidebar.slider("PctExtHyperlinks (0.0 to 1.0)", min_value=0.0, max_value=1.0, value=0.1)
pct_ext_resource_urls = st.sidebar.slider("PctExtResourceUrls (0.0 to 1.0)", min_value=0.0, max_value=1.0, value=0.1)

# Categorical/Binary structural toggles
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌐 Domain & Protocol Features")
    ip_address = st.selectbox("Contains Raw IP Address?", [0, 1], help="1 if domain uses direct IP instead of standard domain name.")
    no_https = st.selectbox("Missing HTTPS Protocol?", [0, 1], help="1 if URL does not use secure HTTPS.")
    at_symbol = st.selectbox("Contains '@' Symbol?", [0, 1])

with col2:
    st.subheader("🔒 Form & Security Indicators")
    insecure_forms = st.selectbox("Insecure Submission Forms?", [0, 1])
    frequent_domain_mismatch = st.selectbox("Domain Mismatch Detected?", [0, 1])
    num_sensitive_words = st.number_input("NumSensitiveWords (e.g., 'secure', 'bank', 'login')", min_value=0, max_value=10, value=0)

# Build feature array matching training schema (48 features baseline)
# Default values set to median zeros for unspecified columns to prevent dimension mismatch
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

# Prediction Execution
st.markdown("---")
if st.button("🚀 Analyze URL Security"):
    input_df = pd.DataFrame([raw_feature_dict])
    
    try:
        # Step 1: Preprocessing using stored Scaler
        scaled_features = scaler.transform(input_df)
        
        # Step 2: Class Prediction
        prediction = model.predict(scaled_features)[0]
        prediction_proba = model.predict_proba(scaled_features)[0]
        
        # Output Display
        st.subheader("🔍 Analysis Result")
        
        if prediction == 1:
            st.error(f"⚠️ **WARNING: Phishing URL Detected!**\n\nProbability of Phishing: **{prediction_proba[1]*100:.2f}%**")
        else:
            st.success(f"✅ **SAFE: Legitimate Website URL.**\n\nProbability of Legitimate Status: **{prediction_proba[0]*100:.2f}%**")

        # Metric Details
        mcol1, mcol2 = st.columns(2)
        mcol1.metric("Legitimate Probability", f"{prediction_proba[0]*100:.1f}%")
        mcol2.metric("Phishing Probability", f"{prediction_proba[1]*100:.1f}%")
        
    except Exception as err:
        st.error(f"Error during execution pipeline: {str(err)}")
