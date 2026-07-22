# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np

# # 1. Page Configuration
# st.set_page_config(page_title="Phishing Web Classifier", layout="wide")

# st.title("🛡️ Phishing Website Detection App")
# st.write("Enter the website feature values below or upload a CSV file to test for phishing.")

# # 2. Load the trained Decision Tree Model Safely
# @st.cache_resource
# def load_model():
#     # Make sure your model file is in the same directory or provide the correct path
#     try:
#         model = joblib.load("phishing_model.pkl") # Replace with your model file name if different
#         return model
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         return None

# model = load_model()

# # List of all 47 features exactly matching the training data order
# features_list = [
#     'NumDots', 'SubdomainLevel', 'PathLevel', 'UrlLength', 'NumDash', 'NumDashInHostname', 
#     'AtSymbol', 'TildeSymbol', 'NumUnderscore', 'NumPercent', 'NumQueryComponents', 
#     'NumAmpersand', 'NumHash', 'NumNumericChars', 'NoHttps', 'RandomString', 'IpAddress', 
#     'DomainInSubdomains', 'DomainInPaths', 'HostnameLength', 'PathLength', 'QueryLength', 
#     'DoubleSlashInPath', 'NumSensitiveWords', 'EmbeddedBrandName', 'PctExtHyperlinks', 
#     'PctExtResourceUrls', 'ExtFavicon', 'InsecureForms', 'RelativeFormAction', 'ExtFormAction', 
#     'AbnormalFormAction', 'PctNullSelfRedirectHyperlinks', 'FrequentDomainNameMismatch', 
#     'FakeLinkInStatusBar', 'RightClickDisabled', 'PopUpWindow', 'SubmitInfoToEmail', 
#     'IframeOrFrame', 'MissingTitle', 'ImagesOnlyInForm', 'SubdomainLevelRT', 'UrlLengthRT', 
#     'PctExtResourceUrlsRT', 'AbnormalExtFormActionR', 'ExtMetaScriptLinkRT', 'PctExtNullSelfRedirectHyperlinksRT'
# ]

# if model is not None:
    
#     # Tabs for Single Input and Batch Input via CSV
#     tab1, tab2 = st.tabs(["📄 Single URL Prediction", "📁 Batch Prediction (CSV Upload)"])
    
#     with tab1:
#         st.subheader("Manual Feature Input")
#         st.write("Provide values for the key URL and HTML features:")
        
#         # Creating a dynamic layout for 47 inputs using columns to keep it clean
#         input_data = {}
#         cols = st.columns(4) # 4 columns layout
        
#         for i, feature in enumerate(features_list):
#             with cols[i % 4]:
#                 # Automatically assigns float input for Pct columns, integer for others
#                 if 'Pct' in feature:
#                     input_data[feature] = st.number_input(f"{feature}", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
#                 else:
#                     input_data[feature] = st.number_input(f"{feature}", min_value=0, value=0, step=1)
                    
#         if st.button("Predict URL Status", type="primary"):
#             # Convert inputs to DataFrame with explicit feature names to prevent scikit-learn warnings/errors
#             features_df = pd.DataFrame([input_data], columns=features_list)
            
#             prediction = model.predict(features_df)
            
#             st.write("---")
#             if prediction[0] == 1:
#                 st.success("✅ This Website appears to be **Legitimate**.")
#             else:
#                 st.error("🚨 Warning! This Website appears to be a **Phishing Scam**.")

#     with tab2:
#         st.subheader("Upload CSV File for Bulk Prediction")
#         st.write("Make sure your CSV contains the required 47 feature columns. (Any extra column like 'id' will be safely ignored).")
        
#         uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        
#         if uploaded_file is not None:
#             df = pd.read_csv(uploaded_file)
            
#             # Check if all required features exist in uploaded CSV
#             missing_cols = [col for col in features_list if col not in df.columns]
            
#             if missing_cols:
#                 st.error(f"The uploaded CSV is missing the following required columns: {missing_cols}")
#             else:
#                 # Keep only the trained columns and rearrange them in the correct training order
#                 test_df = df[features_list]
                
#                 # Perform batch prediction
#                 batch_predictions = model.predict(test_df)
                
#                 # Append results to the original uploaded dataframe
#                 df['Prediction_Label'] = batch_predictions
#                 df['Prediction_Result'] = np.where(df['Prediction_Label'] == 1, 'Legitimate', 'Phishing')
                
#                 st.write("### Prediction Preview:")
#                 st.dataframe(df[['Prediction_Result'] + features_list].head(10))
                
#                 # Download button for the results
#                 output_csv = df.to_csv(index=False).encode('utf-8')
#                 st.download_button(
#                     label="📥 Download Predictions CSV",
#                     data=output_csv,
#                     file_name="phishing_predictions_output.csv",
#                     mime="text/csv"
#                 )
# else:
#     st.info("💡 Please ensure that your serialized model file is correctly saved as 'phishing_model.pkl' in the application folder.")











import streamlit as st
import pandas as pd
import joblib
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

# 1. Page Configuration for Production
st.set_page_config(
    page_title="Phishing Web Classifier", 
    layout="wide", 
    page_icon="🛡️"
)

st.title("🛡️ Real-Time Phishing Detection & Analysis App")
st.write("Scan URLs or upload CSV files to classify websites using Machine Learning.")

# 2. Load the trained Model Safely
@st.cache_resource
def load_model():
    try:
        model = joblib.load("phishing_model.pkl") 
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
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

# 3. REAL-TIME URL & HTML FEATURE EXTRACTOR
def extract_real_features(url):
    features = {f: 0 for f in features_list}
    url_clean = url.strip()
    
    if not url_clean.startswith(('http://', 'https://')):
        url_clean = 'http://' + url_clean
        
    try:
        parsed_url = urlparse(url_clean)
        hostname = parsed_url.hostname if parsed_url.hostname else ""
        path = parsed_url.path if parsed_url.path else ""
        query = parsed_url.query if parsed_url.query else ""
    except Exception:
        hostname, path, query = "", "", ""

    # Basic Structural Calculations
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
    features['NoHttps'] = 1 if url_clean.startswith('http://') else 0
    
    features['RandomString'] = 1 if re.search(r'[a-zA-Z0-9]{10,}', url_clean) else 0
    ip_match = re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', hostname)
    features['IpAddress'] = 1 if ip_match else 0
    
    features['DomainInSubdomains'] = 1 if hostname and hostname.count('.') > 2 else 0
    features['DomainInPaths'] = 1 if hostname and hostname in path else 0
    features['HostnameLength'] = len(hostname)
    features['PathLength'] = len(path)
    features['QueryLength'] = len(query)
    features['DoubleSlashInPath'] = 1 if '//' in path else 0
    
    sensitive_words = ['login', 'verify', 'secure', 'update', 'banking', 'account', 'signin', 'fb', 'verification', 'password']
    features['NumSensitiveWords'] = sum(1 for word in sensitive_words if word in url_clean.lower())
    
    brands = ['paypal', 'ebay', 'amazon', 'facebook', 'google', 'microsoft', 'netflix', 'apple', 'instagram']
    features['EmbeddedBrandName'] = 1 if any(brand in hostname for brand in brands) else 0

    features['SubdomainLevelRT'] = 1 if features['SubdomainLevel'] <= 1 else (-1 if features['SubdomainLevel'] >= 3 else 0)
    features['UrlLengthRT'] = 1 if len(url_clean) < 54 else (-1 if len(url_clean) > 75 else 0)

    # LIVE FETCH HTML/DOM DATA
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url_clean, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # HTML Elements Extraction
        features['MissingTitle'] = 0 if (soup.title and soup.title.string) else 1
        features['IframeOrFrame'] = 1 if (soup.find('iframe') or soup.find('frame')) else 0
        
        # Check Form Actions & Input Fields
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
                    
        # Hyperlink domain checks
        links = soup.find_all('a', href=True)
        if links:
            ext_links = [l['href'] for l in links if l['href'].startswith('http') and hostname not in l['href']]
            features['PctExtHyperlinks'] = len(ext_links) / len(links)
            
    except Exception:
        # Graceful Fallback if site blocks crawling/timeout occurs
        pass

    return features

# --- PRODUCTION UI LOGIC ---
if model is not None:
    
    tab1, tab2, tab3 = st.tabs([
        "🔗 Real-Time URL Scan", 
        "✍️ Manual Feature Input", 
        "📁 Batch Prediction (CSV)"
    ])
    
    # ---------------- TAB 1: PASTE URL ----------------
    with tab1:
        st.subheader("Instant URL Analysis")
        st.write("Paste a link to scan its structural properties and HTML content in real-time.")
        
        user_url = st.text_input("Enter Website Address", placeholder="https://example.com", key="prod_url")
        
        if st.button("Analyze Link", type="primary", key="scan_btn"):
            if not user_url.strip():
                st.warning("⚠️ Please provide a URL.")
            else:
                with st.spinner("Analyzing web properties..."):
                    extracted_data = extract_real_features(user_url)
                    features_df = pd.DataFrame([extracted_data], columns=features_list)
                    
                    prediction = model.predict(features_df)
                    
                    st.write("---")
                    # Check mapping (1 = Legitimate, 0 = Phishing)
                    if prediction[0] == 1:
                        st.success("✅ This Website appears to be **Legitimate**.")
                    else:
                        st.error("🚨 Warning! This Website appears to be a **Phishing Scam**.")
                        
                with st.expander("📊 View Extracted Features"):
                    st.json({k: v for k, v in extracted_data.items() if v != 0})

    # ---------------- TAB 2: MANUAL FEATURE INPUT ----------------
    with tab2:
        st.subheader("Manual Feature Input")
        input_data = {}
        cols = st.columns(4)
        
        for i, feature in enumerate(features_list):
            with cols[i % 4]:
                if 'Pct' in feature:
                    input_data[feature] = st.number_input(f"{feature}", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key=f"m_{feature}")
                else:
                    input_data[feature] = st.number_input(f"{feature}", min_value=0, value=0, step=1, key=f"m_{feature}")
                    
        if st.button("Predict via Manual Inputs", key="manual_btn"):
            features_df = pd.DataFrame([input_data], columns=features_list)
            prediction = model.predict(features_df)
            
            st.write("---")
            if prediction[0] == 1:
                st.success("✅ Legitimate Website")
            else:
                st.error("🚨 Phishing Website")

    # ---------------- TAB 3: BATCH PREDICTION ----------------
    with tab3:
        st.subheader("Batch CSV Prediction")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_uploader")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            missing_cols = [col for col in features_list if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing columns in CSV: {missing_cols}")
            else:
                test_df = df[features_list]
                batch_predictions = model.predict(test_df)
                
                df['Prediction_Label'] = batch_predictions
                df['Prediction_Result'] = np.where(df['Prediction_Label'] == 1, 'Legitimate', 'Phishing')
                
                st.dataframe(df[['Prediction_Result'] + features_list].head(10))
                
                output_csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Output CSV",
                    data=output_csv,
                    file_name="phishing_results.csv",
                    mime="text/csv",
                    key="download_btn"
                )
else:
    st.info("💡 Ensure `phishing_model.pkl` is saved in the working directory.")