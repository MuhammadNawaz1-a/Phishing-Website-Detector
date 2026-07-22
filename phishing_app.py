import re
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# 1. Page Configuration
st.set_page_config(
    page_title="Phishing Web Classifier", 
    layout="wide", 
    page_icon="🛡️"
)

st.title("🛡️ Real-Time Phishing Website Detector")
st.write("Scan URLs, manually input feature vectors, or process batch CSV files using Machine Learning.")

# 2. Safe Model Loading
@st.cache_resource
def load_model():
    try:
        model = joblib.load("phishing_model.pkl") 
        return model
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        return None

model = load_model()

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

# 3. Enhanced Real-Time Feature Extractor Function
def extract_real_features(url):
    features = {f: 0 for f in features_list}
    url_clean = url.strip()
    
    if not url_clean.startswith(('http://', 'https://')):
        url_clean = 'https://' + url_clean  # Default to https for modern websites
        
    try:
        parsed_url = urlparse(url_clean)
        hostname = parsed_url.hostname if parsed_url.hostname else ""
        path = parsed_url.path if parsed_url.path else ""
        query = parsed_url.query if parsed_url.query else ""
    except Exception:
        hostname, path, query = "", "", ""

    # Structural feature calculations
    features['NumDots'] = url_clean.count('.')
    features['SubdomainLevel'] = max(0, hostname.count('.') - 1) if hostname else 0
    features['PathLevel'] = path.count('/') if path else 0
    features['UrlLength'] = len(url_clean)
    features['NumDash'] = url_clean.count('-')
    features['NumDashInHostname'] = hostname.count('-') if hostname else 0
    features['AtSymbol'] = 1 if '@' in url_clean else 0
    features['TildeSymbol'] = 1 if '~' in url_clean else 0
    features['NumUnderscore'] = url_clean.count('_')
    features['NumPercent'] = url_clean.count('%')
    features['NumQueryComponents'] = len(query.split('&')) if query else 0
    features['NumAmpersand'] = url_clean.count('&')
    features['NumHash'] = url_clean.count('#')
    features['NumNumericChars'] = sum(c.isdigit() for c in url_clean)
    features['NoHttps'] = 1 if url_clean.startswith('http://') and not url_clean.startswith('https://') else 0
    
    features['RandomString'] = 1 if re.search(r'[a-zA-Z0-9]{10,}', url_clean) else 0
    ip_match = re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', hostname)
    features['IpAddress'] = 1 if ip_match else 0
    
    features['DomainInSubdomains'] = 1 if hostname and hostname.count('.') > 2 else 0
    features['DomainInPaths'] = 1 if hostname and hostname in path else 0
    features['HostnameLength'] = len(hostname)
    features['PathLength'] = len(path)
    features['QueryLength'] = len(query)
    features['DoubleSlashInPath'] = 1 if '//' in path else 0
    
    sensitive_words = ['login', 'verify', 'secure', 'update', 'banking', 'account', 'signin', 'password']
    features['NumSensitiveWords'] = sum(1 for word in sensitive_words if word in url_clean.lower())
    
    brands = ['paypal', 'ebay', 'amazon', 'facebook', 'google', 'microsoft', 'netflix', 'apple']
    features['EmbeddedBrandName'] = 1 if any(brand in hostname for brand in brands) else 0

    features['SubdomainLevelRT'] = 1 if features['SubdomainLevel'] <= 1 else (-1 if features['SubdomainLevel'] >= 3 else 0)
    features['UrlLengthRT'] = 1 if len(url_clean) < 54 else (-1 if len(url_clean) > 75 else 0)

    # DOM/HTML parsing with Browser Headers & Timeout Fallback
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
        }
        response = requests.get(url_clean, headers=headers, timeout=5, verify=False)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            features['MissingTitle'] = 0 if (soup.title and soup.title.string) else 1
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
                features['PctExtHyperlinks'] = len(ext_links) / len(links)
        else:
            # Fallback if site blocks scraping
            features['MissingTitle'] = 0
    except Exception:
        # Graceful degradation if scraping fails
        features['MissingTitle'] = 0

    return features

# 4. Interface Application Logic
if model is not None:
    tab1, tab2, tab3 = st.tabs(["🔗 Real-Time URL Scan", "✍️ Manual Feature Input", "📁 Batch Prediction (CSV)"])
    
    # --- Tab 1: Real-Time Scan ---
    with tab1:
        st.subheader("Instant URL Analysis")
        user_url = st.text_input("Enter Website URL", placeholder="https://google.com", key="scan_url_input")
        
        if st.button("Analyze URL", type="primary", key="scan_btn"):
            if not user_url.strip():
                st.warning("Please enter a valid website address.")
            else:
                with st.spinner("Analyzing web features..."):
                    extracted_data = extract_real_features(user_url)
                    features_df = pd.DataFrame([extracted_data], columns=features_list)
                    prediction = model.predict(features_df)
                    
                    st.write("---")
                    if prediction[0] == 1:
                        st.success("✅ This Website appears to be **Legitimate**.")
                    else:
                        st.error("🚨 Warning! This Website appears to be a **Phishing Scam**.")

    # --- Tab 2 & Tab 3 code remains same ---
