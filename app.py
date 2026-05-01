import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import LinearRegression
from transformer import flexible_transform

# --- PAGE CONFIG ---
st.set_page_config(page_title="Real Estate AI", layout="wide", page_icon="🏘️")

# --- CSS OVERRIDES ---
st.markdown("""
<style>
    /* 1. THEME RESET */
    [data-testid="stHeader"] { background-color: rgba(0,0,0,0) !important; }
    [data-testid="stDecoration"] { display: none !important; }
    .stApp, [data-testid="stAppViewContainer"] { background-color: #0f172a !important; }

    /* 2. SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: rgba(30, 41, 59, 0.9) !important;
        backdrop-filter: blur(20px);
    }
    section[data-testid="stSidebar"] h3 { color: #60a5fa !important; font-weight: 800 !important; }

    /* 3. UPLOADED FILE NAME VISIBILITY */
    [data-testid="stFileUploaderFileName"] {
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    /* 4. FORCE BLACK TEXT IN STATUS WIDGET */
    /* Target the container background to ensure black text is readable */
    div[data-testid="stStatusWidget"] {
        background-color: #ffffff !important; 
        border: 2px solid #60a5fa !important;
        border-radius: 10px !important;
    }
    
    /* Target the summary text "Training Model..." and child text */
    div[data-testid="stStatusWidget"] summary,
    div[data-testid="stStatusWidget"] summary span,
    div[data-testid="stStatusWidget"] label,
    div[data-testid="stStatusWidget"] p,
    div[data-testid="stStatusWidget"] div {
        color: #000000 !important; /* FORCING BLACK */
        font-weight: 700 !important;
    }
    
    /* Spinner color for consistency */
    .stSpinner i { border-top-color: #000000 !important; }

    /* 5. UI COMPONENTS */
    div[data-testid="stVerticalBlock"] > div:has(div.stNumberInput) {
        background-color: rgba(30, 41, 59, 0.5) !important;
        backdrop-filter: blur(12px);
        padding: 25px;
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    .stButton>button {
        background: linear-gradient(180deg, #22c55e 0%, #16a34a 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
    }

    .prediction-container {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.85) 0%, rgba(22, 163, 74, 0.85) 100%);
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        margin-top: 25px;
    }

    .prediction-value {
        color: #ffffff !important;
        font-size: 3.8rem !important;
        font-weight: 800 !important;
    }

    label, p, .stSelectbox, .stNumberInput { color: #cbd5e1 !important; }
</style>
""", unsafe_allow_html=True)

# --- TRAINING ENGINE ---
def train_model(df):
    if 'size' in df.columns:
        df['bhk'] = df['size'].apply(lambda x: int(str(x).split(' ')[0]) if pd.notnull(x) else 2)
    
    def clean_sqft(x):
        try:
            tokens = str(x).split('-')
            if len(tokens) == 2: return (float(tokens[0]) + float(tokens[1]))/2
            return float(x)
        except: return None
        
    if 'total_sqft' in df.columns:
        df['total_sqft'] = df['total_sqft'].apply(clean_sqft)
    
    df = df.dropna(subset=['total_sqft', 'price', 'bath', 'location'])
    
    if df.empty:
        return "Error: Dataset empty after cleaning."

    dummies = pd.get_dummies(df.location)
    X = pd.concat([df[['total_sqft', 'bath', 'bhk']], dummies], axis='columns')
    y = df.price
    
    model = LinearRegression()
    model.fit(X.values, y)
    return model, X.columns

# --- UI APP LOGIC ---
def main():
    st.sidebar.markdown("### 🏘️ AI Control Panel")
    
    if 'target_area' not in st.session_state:
        st.session_state.target_area = "Bengaluru (Pre-trained)"
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = "Predict Price"

    pkl_path = os.path.join(os.path.dirname(__file__), 'house_price_model.pkl')
    m_key = f"model_{st.session_state.target_area}"
    
    if m_key not in st.session_state and os.path.exists(pkl_path):
        with st.sidebar.spinner("Initializing..."):
            try:
                with open(pkl_path, 'rb') as f:
                    payload = pickle.load(f)
                    st.session_state[m_key] = payload['model'] if isinstance(payload, dict) else payload
                    st.session_state[f"cols_{st.session_state.target_area}"] = payload['columns'] if isinstance(payload, dict) else []
            except: pass

    if st.sidebar.button("📤 Train Custom Model"):
        st.session_state.current_mode = "Update Data"
        st.rerun()

    mode = st.session_state.current_mode

    if mode == "Update Data":
        st.markdown("<h1 style='color: #60a5fa;'>📊 Data Training Studio</h1>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload your data", type="csv")
        
        if uploaded_file:
            new_area = uploaded_file.name.replace(".csv", "")
            raw_df = pd.read_csv(uploaded_file)
            
            if st.button(f"🚀 Begin Training for {new_area}"):
                # Text forced to black via CSS selector div[data-testid="stStatusWidget"] summary
                with st.status(f"Training Model...", expanded=True) as status:
                    st.write("Processing data...")
                    clean_df, error = flexible_transform(raw_df)
                    if error:
                        st.error(error)
                        status.update(label="Training Failed!", state="error")
                    else:
                        st.write("Building weights...")
                        result = train_model(clean_df)
                        if not isinstance(result, str):
                            model, cols = result
                            st.session_state.target_area = new_area
                            st.session_state[f'model_{new_area}'] = model
                            st.session_state[f'cols_{new_area}'] = cols
                            st.session_state.current_mode = "Predict Price"
                            status.update(label="Training Complete!", state="complete")
                            st.rerun()
        
        if st.button("Cancel"):
            st.session_state.current_mode = "Predict Price"
            st.rerun()

    elif mode == "Predict Price":
        st.markdown(f"<h1 style='color: #60a5fa;'>🏘️ Real Estate Price Prediction</h1>", unsafe_allow_html=True)
        
        area = st.session_state.target_area
        m_key, c_key = f'model_{area}', f'cols_{area}'
        
        if m_key in st.session_state:
            model = st.session_state[m_key]
            columns = st.session_state[c_key]
            locations = [c for c in columns if c not in ['total_sqft', 'bath', 'bhk']]
            
            col1, col2 = st.columns(2)
            with col1:
                loc = st.selectbox("Neighborhood", locations, 
                                 format_func=lambda x: x.replace("location_", ""))
                sqft = st.number_input("Area (Sqft)", min_value=100, value=1200)
            with col2:
                bhk = st.slider("BHK", 1, 10, 2)
                bath = st.slider("Bathrooms", 1, 10, 2)
            
            if st.button("Get Price Estimate"):
                with st.spinner("Calculating..."):
                    x = np.zeros(len(columns))
                    x[0], x[1], x[2] = sqft, bath, bhk
                    if loc in columns:
                        loc_idx = np.where(columns == loc)[0][0]
                        x[loc_idx] = 1
                    
                    prediction = model.predict([x])[0]
                    st.markdown(f"""
                        <div class="prediction-container">
                            <div class="prediction-value">₹ {round(prediction, 2)} Lakhs</div>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("AI model standby. Load a dataset to begin.")

if __name__ == '__main__':
    main()