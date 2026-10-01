import joblib
import streamlit as st
import os

# ─── Page Config ───
st.set_page_config(page_title="Spam Detection App", page_icon="", layout="wide")

# ─── 1. Load Model & Data ───
base_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Load your saved files
model = joblib.load(os.path.join(base_dir, 'spam_model.joblib'))
vectorizer = joblib.load(os.path.join(base_dir, 'vectorizer.joblib'))

# 2. Create the Sidebar for Model Info
st.sidebar.title("🤖 Model Information")
st.sidebar.markdown("This app uses Naive Bayes and NLP to detect spam.")
st.sidebar.metric(label="Model Accuracy", value="98%") # Based on your SMOTE results!

# 3. Main Page Header
st.title(" Spam Detection App")
st.write("Enter an email or SMS message below to check if it's spam.")

# 4. User Input
message = st.text_area("Enter message", height=150, placeholder="Type or paste your message here...")

# 5. Prediction Logic
if st.button('Predict'):
    if message:
        # Transform the text using your loaded vectorizer
        transformed = vectorizer.transform([message])
        
        # Get the prediction (0 or 1)
        prediction = model.predict(transformed)
        
        # Get the probability scores for confidence
        probabilities = model.predict_proba(transformed)[0]
        confidence = max(probabilities) # Get the highest probability

        # 6. Display the Results
        if prediction[0] == 1:
            st.warning("⚠️ This message is likely SPAM!")
        else:
            st.success("✅ This message is NOT Spam (Ham).")

        # Show the confidence score visually
        st.metric(label="Confidence Score", value=f"{confidence * 100:.2f}%")
        st.progress(confidence)