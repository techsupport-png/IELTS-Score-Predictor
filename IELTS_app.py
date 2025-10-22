import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Page config
st.set_page_config(
    page_title="IELTS Score Predictor",
    page_icon="🎓",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    rf_model = joblib.load('rf_model.pkl')
    xgb_model = joblib.load('xgb_model.pkl')
    nn_model = load_model('nn_model.h5', compile=False)
    le_board = joblib.load('le_board.pkl')
    le_gender = joblib.load('le_gender.pkl')
    scaler = joblib.load('scaler.pkl')
    return rf_model, xgb_model, nn_model, le_board, le_gender, scaler

rf_model, xgb_model, nn_model, le_board, le_gender, scaler = load_models()

# Header
st.title("🎓 IELTS Score Predictor")
st.markdown("### Predict your IELTS band score based on your academic performance")
st.markdown("---")

# All Indian Education Boards
boards = [
    'CBSE', 'ICSE', 'CISCE', 'IB', 'NIOS',
    'Maharashtra State Board', 'Tamil Nadu State Board', 'Karnataka State Board',
    'Andhra Pradesh State Board', 'Telangana State Board', 'Kerala State Board',
    'West Bengal State Board', 'Gujarat State Board', 'Rajasthan State Board',
    'Madhya Pradesh State Board', 'Uttar Pradesh State Board', 'Bihar State Board',
    'Odisha State Board', 'Punjab State Board', 'Haryana State Board',
    'Jharkhand State Board', 'Chhattisgarh State Board', 'Assam State Board',
    'Jammu and Kashmir State Board', 'Himachal Pradesh State Board',
    'Uttarakhand State Board', 'Goa State Board', 'Tripura State Board',
    'Meghalaya State Board', 'Manipur State Board', 'Nagaland State Board',
    'Mizoram State Board', 'Arunachal Pradesh State Board', 'Sikkim State Board'
]

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Enter Your Details")
    
    # Input fields
    board = st.selectbox("Board of Education", boards, index=0)
    
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    
    percentage = st.number_input(
        "12th Grade Percentage (%)",
        min_value=35.0,
        max_value=100.0,
        value=75.0,
        step=0.1,
        help="Enter your 12th grade percentage"
    )
    
    english_score = st.number_input(
        "12th English Score (out of 100)",
        min_value=35,
        max_value=100,
        value=80,
        step=1,
        help="Enter your 12th grade English subject score"
    )
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔮 Predict My IELTS Score", type="primary", use_container_width=True):
        # Prepare input
        board_encoded = le_board.transform([board])[0]
        gender_encoded = le_gender.transform([gender])[0]
        
        # Create feature array
        features = np.array([[
            board_encoded,
            gender_encoded,
            percentage,
            english_score
        ]])
        
        # Scale numerical features
        features_scaled = features.copy()
        features_scaled[:, 2:4] = scaler.transform(features[:, 2:4])
        
        # Get predictions from all models
        rf_pred = rf_model.predict(features_scaled)[0]
        xgb_pred = xgb_model.predict(features_scaled)[0]
        nn_pred = nn_model.predict(features_scaled, verbose=0)[0][0]
        
        # Use XGBoost (best model) and round to nearest 0.5
        predicted_score = round(xgb_pred * 2) / 2
        predicted_score = max(5.0, min(9.0, predicted_score))  # Clamp between 5.0 and 9.0
        
        # Store in session state
        st.session_state.prediction = {
            'score': predicted_score,
            'rf': round(rf_pred * 2) / 2,
            'xgb': round(xgb_pred * 2) / 2,
            'nn': round(nn_pred * 2) / 2
        }

with col2:
    st.subheader("🎯 Prediction Results")
    
    if 'prediction' in st.session_state:
        pred = st.session_state.prediction
        
        # Main prediction card
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 40px; 
                    border-radius: 20px; 
                    text-align: center;
                    color: white;
                    margin-bottom: 20px;'>
            <h1 style='font-size: 80px; margin: 0; font-weight: bold;'>{pred['score']:.1f}</h1>
            <h3 style='margin: 10px 0;'>Predicted IELTS Band Score</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Band level
        if pred['score'] >= 8.5:
            level = "Expert User"
            color = "#9333ea"
        elif pred['score'] >= 7.5:
            level = "Very Good User"
            color = "#2563eb"
        elif pred['score'] >= 6.5:
            level = "Good User"
            color = "#16a34a"
        elif pred['score'] >= 5.5:
            level = "Modest User"
            color = "#ca8a04"
        else:
            level = "Limited User"
            color = "#ea580c"
        
        st.markdown(f"""
        <div style='background-color: {color}20; 
                    border-left: 5px solid {color}; 
                    padding: 15px; 
                    border-radius: 10px;
                    margin-bottom: 20px;'>
            <h3 style='color: {color}; margin: 0;'>{level}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Model info
        st.info("🎯 **Predicted using XGBoost Model** - Our best performing model with 79.7% accuracy (R² Score)")
        
        # Score range
        score_min = max(5.0, pred['score'] - 0.5)
        score_max = min(9.0, pred['score'] + 0.5)
        st.success(f"📊 **Expected Score Range:** {score_min:.1f} - {score_max:.1f}")
        
    else:
        st.info("👈 Enter your details and click 'Predict' to see your estimated IELTS score!")
        
        # IELTS Band levels guide
        st.markdown("### 📚 IELTS Band Levels")
        
        band_levels = {
            "9.0": ("Expert User", "Full operational command of the language"),
            "8.0-8.5": ("Very Good User", "Fully operational command with occasional inaccuracies"),
            "7.0-7.5": ("Good User", "Operational command with occasional inaccuracies"),
            "6.0-6.5": ("Competent User", "Generally effective command"),
            "5.0-5.5": ("Modest User", "Partial command of the language"),
        }
        
        for band, (level, desc) in band_levels.items():
            st.markdown(f"**{band}** - {level}")
            st.caption(desc)
            st.markdown("")

# Footer
st.markdown("---")
col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.metric("Model Accuracy", "79.7%", help="R² Score on test data")

with col_info2:
    st.metric("Training Data", "30,000+", help="Student records used for training")

with col_info3:
    st.metric("Boards Supported", "34", help="Indian education boards")

st.markdown("---")
st.caption("Powered by Machine Learning • Random Forest + XGBoost + Neural Networks")