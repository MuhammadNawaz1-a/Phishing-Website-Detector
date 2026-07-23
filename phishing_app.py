import streamlit as st
import pandas as pd
import joblib
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# 1. Page Configuration
st.set_page_config(
    page_title="Phishing Web Classifier", 
    layout="wide", 
    page_icon="🛡️"
)

st.title("🛡️ Phishing Website Detection Application")
st.write("Scan web links, input manual feature values, or process batch CSV files using Machine Learning.")

# 2. Model Loader with Resource Caching
@st.cache_resource
def load_model():
    try:
        model = joblib.load("phishing_model.pkl") 
        return model
    except Exception as e:
        st.error(f"Error loading model file 'phishing_model.pkl': {e}")
        return None

model = load_model()

# List of all 47 features matching the model training order exactly
features_list = [
    'NumDots', 'SubdomainLevel', 'PathLevel', 'UrlLength', 'NumDash', 'NumDashInHostname', 
    'AtSymbol', 'TildeSymbol', 'NumUnderscore', 'NumPercent', 'NumQueryComponents', 
    'NumAmpersand', 'NumHash', 'NumNumericChars', 'NoHttps', 'RandomString', 'IpAddress', 
    'DomainInSubdomains', 'DomainInPaths', 'HostnameLength', 'PathLength', 'QueryLength', 
    'DoubleSlashInPath', 'NumSensitiveWords', 'EmbeddedBrandName', 'PctExtHyperlinks', 
    'PctExtResourceUrls', 'ExtFavicon', 'InsecureForms', 'RelativeFormAction', 'ExtFormAction', 
    'AbnormalFormAction', 'PctNullSelfRedirectHyperlinks', 'FrequentDomainNameMismatch', 
    'FakeLinkInStatusBar', 'RightClickDisabled', 'PopUpWindow', 'SubmitInfoToEmail', 
    'IframeOrFrame', 'MissingTitle', 'ImagesOnlyInForm', 'SubdomainLevelRT', 'UrlLengthRT', 
    'PctExtResourceUrlsRT', 'AbnormalExtFormActionR', 'ExtMetaScriptLinkRT', 'PctExtNullSelfRedirectHyperlinksRT'
]

# 3. Real-Time Feature Extraction Function
def extract_real_features(url):
    features = {f: 0 for f in features_list}
    url_clean = url.strip()
    
    if not url_clean.startswith(('http://', 'https://')):
        url_clean = 'https://' + url_clean
        
    try:
        parsed_url = urlparse(url_clean)
        hostname = parsed_url.hostname if parsed_url.hostname else ""
        path = parsed_url.path if parsed_url.path else ""
        query = parsed_url.query if parsed_url.query else ""
    except Exception:
        hostname, path, query = "", "", ""

    # Structural feature calculations
    features['NumDots'] = url_clean.count('.')
    
    # Subdomain calculation standard
    if hostname:
        parts = hostname.split('.')
        features['SubdomainLevel'] = max(0, len(parts) - 2) if len(parts) > 2 else 0
    else:
        features['SubdomainLevel'] = 0

    features['PathLevel'] = path.count('/') if path else 0
    features['UrlLength'] = len(url_clean)
    features['NumDash'] = url_clean.count('-')
    features['NumDashInHostname'] = hostname.count('-') if hostname else 0
    features['AtSymbol'] = 1 if '@' in url_clean else 0
    features['TildeSymbol'] = 1 if '~' in url_clean else 0
    features['NumUnderscore'] = url_clean.count('_')
    features['NumPercent'] = url_clean.count('%')
    features['NumQueryComponents'] = len(query.split('&')) if query and query.strip() else 0
    features['NumAmpersand'] = url_clean.count('&')
    features['NumHash'] = url_clean.count('#')
    features['NumNumericChars'] = sum(c.isdigit() for c in url_clean)
    features['NoHttps'] = 1 if url_clean.startswith('http://') else 0
    
    features['RandomString'] = 1 if re.search(r'[a-zA-Z0-9]{12,}', url_clean) else 0
    features['IpAddress'] = 1 if re.search(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', hostname) else 0
    
    features['DomainInSubdomains'] = 1 if hostname and features['SubdomainLevel'] > 1 else 0
    features['DomainInPaths'] = 1 if hostname and len(path) > 1 and hostname in path else 0
    features['HostnameLength'] = len(hostname)
    features['PathLength'] = len(path)
    features['QueryLength'] = len(query)
    features['DoubleSlashInPath'] = 1 if '//' in path else 0
    
    sensitive_words = ['login', 'verify', 'secure', 'update', 'banking', 'account', 'signin', 'password', 'confirm']
    features['NumSensitiveWords'] = sum(1 for word in sensitive_words if word in url_clean.lower())
    
    brands = ['paypal', 'ebay', 'amazon', 'facebook', 'google', 'microsoft', 'netflix', 'apple', 'steam']
    
    # Check if brand is embedded into path or third-level subdomain rather than primary domain
    features['EmbeddedBrandName'] = 0
    if hostname:
        main_domain = ".".join(hostname.split('.')[-2:])
        if any(b in url_clean.lower() and b not in main_domain for b in brands):
            features['EmbeddedBrandName'] = 1

    features['SubdomainLevelRT'] = 1 if features['SubdomainLevel'] <= 1 else (-1 if features['SubdomainLevel'] >= 3 else 0)
    features['UrlLengthRT'] = 1 if len(url_clean) < 54 else (-1 if len(url_clean) > 75 else 0)

    # Live HTML fetching with graceful error handling
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url_clean, headers=headers, timeout=3, allow_redirects=True)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        features['MissingTitle'] = 0 if (soup.title and soup.title.string and soup.title.string.strip()) else 1
        features['IframeOrFrame'] = 1 if (soup.find('iframe') or soup.find('frame')) else 0
        
        forms = soup.find_all('form')
        if forms:
            for form in forms:
                action = form.get('action', '')
                if not action:
                    features['PctNullSelfRedirectHyperlinks'] += 1
                elif action.startswith('http') and hostname not in action:
                    features['ExtFormAction'] = 1
                elif not action.startswith('http'):
                    features['RelativeFormAction'] = 1
                
                if 'mailto:' in action.lower():
                    features['SubmitInfoToEmail'] = 1
                    
        links = soup.find_all('a', href=True)
        if links:
            ext_links = [l['href'] for l in links if l['href'].startswith('http') and hostname not in l['href']]
            features['PctExtHyperlinks'] = round(len(ext_links) / len(links), 4)
            
    except Exception:
        # Fallback values for offline or non-responsive links
        features['MissingTitle'] = 0

    return features

# 4. User Interface Tabs
if model is not None:
    
    tab1, tab2, tab3 = st.tabs([
        "🔗 Real-Time Link Scanner", 
        "✍️ Manual Feature Input", 
        "📁 Batch Prediction (CSV Upload)"
    ])
    
    # --- TAB 1: Live Scanner ---
    with tab1:
        st.subheader("Automated Link Analysis")
        st.write("Enter any URL below to automatically extract structural and DOM parameters for analysis.")
        
        user_url = st.text_input("Website URL", placeholder="https://example.com", key="tab1_url")
        
        if st.button("Scan Website", type="primary", key="tab1_btn"):
            clean_input = user_url.strip()
            if not clean_input:
                st.warning("⚠️ Please enter a valid website link first before scanning.")
            else:
                with st.spinner("Analyzing web properties..."):
                    extracted_data = extract_real_features(clean_input)
                    features_df = pd.DataFrame([extracted_data], columns=features_list)
                    
                    prediction = model.predict(features_df)
                    
                    st.write("---")
                    if prediction[0] == 1:
                        st.success(f"✅ **Legitimate Website**: `{clean_input}` appears to be safe.")
                    else:
                        st.error(f"🚨 **Phishing Warning**: `{clean_input}` shows suspicious characteristics.")
                        
                with st.expander("📊 View Extracted Vector Features"):
                    non_zero = {k: v for k, v in extracted_data.items() if v != 0}
                    st.json(non_zero if non_zero else extracted_data)

    # --- TAB 2: Manual Feature Input ---
    with tab2:
        st.subheader("Manual Feature Specification")
        input_data = {}
        cols = st.columns(4)
        
        for i, feature in enumerate(features_list):
            with cols[i % 4]:
                if 'Pct' in feature:
                    input_data[feature] = st.number_input(f"{feature}", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key=f"t2_{feature}")
                else:
                    input_data[feature] = st.number_input(f"{feature}", min_value=0, value=0, step=1, key=f"t2_{feature}")
                    
        if st.button("Run Manual Prediction", type="primary", key="tab2_btn"):
            features_df = pd.DataFrame([input_data], columns=features_list)
            prediction = model.predict(features_df)
            
            st.write("---")
            if prediction[0] == 1:
                st.success("✅ Prediction Result: **Legitimate**")
            else:
                st.error("🚨 Prediction Result: **Phishing**")

    # --- TAB 3: Batch Prediction ---
    with tab3:
        st.subheader("Bulk File Processing")
        uploaded_file = st.file_uploader("Choose CSV Dataset", type=["csv"], key="tab3_csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            missing_cols = [col for col in features_list if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns in dataset: {missing_cols}")
            else:
                test_df = df[features_list]
                batch_predictions = model.predict(test_df)
                
                df['Prediction_Label'] = batch_predictions
                df['Prediction_Result'] = np.where(df['Prediction_Label'] == 1, 'Legitimate', 'Phishing')
                
                st.write("### Prediction Preview:")
                st.dataframe(df[['Prediction_Result'] + features_list].head(10))
                
                output_csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=output_csv,
                    file_name="phishing_predictions_output.csv",
                    mime="text/csv",
                    key="tab3_download"
                )
else:
    st.info("Please make sure `phishing_model.pkl` is located in your project directory.")
