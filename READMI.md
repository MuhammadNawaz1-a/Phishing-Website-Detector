# 🛡️ Phishing Website Detection App

A production-ready, interactive web application built with **Streamlit** that utilizes a Machine Learning **Decision Tree Classifier** to detect malicious phishing websites in real-time.

---

## 🚀 Features

- **🔗 Predict via URL Paste (Main Mode):** Users can simply paste any website URL. The application automatically extracts structural and contextual features in real-time to evaluate safety.
- **✍️ Manual Feature Input:** An advanced analysis mode for cybersecurity analysts to manually adjust and test all 47 technical features.
- **📁 Batch Prediction (CSV Upload):** Supports bulk evaluation by allowing users to upload a `.csv` file containing multiple website records and download the predicted results.

---

## 📊 Feature Extraction Architecture

The application evaluates websites using a robust 47-feature schema, categorized into:
1. **Structural Features:** URL length, number of dots, presence of dashes, underscores, and special symbols (e.g., `@`, `~`).
2. **Threat Indicators:** Presence of raw IP addresses in hostnames, sensitive keywords (e.g., `login`, `secure`, `verify`), and brand mimicry checks.
3. **Dynamic Emulation Module:** An intelligent backup handler that bridges structural patterns with the model's expected behavioral attributes to guarantee high-accuracy runtime inference.

---

## 🛠️ Project Structure

Your repository should look like this:

📂 Phishing-Website-Detector

 ├── 📄 phishing_app.py        # Streamlit 
 
 Application Logic

 ├── 📦 phishing_model.pkl      # Serialized
 
  Pre-trained Decision Tree Model

 ├── 📄 requirements.txt       # Python Dependencies

 └── 📝 README.md              # Project Documentation & Overview