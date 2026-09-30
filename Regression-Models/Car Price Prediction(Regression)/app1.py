import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ─── Page Config ───
st.set_page_config(page_title="Car Price Predictor", page_icon="", layout="wide")

# ─── Custom CSS ───
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1E88E5; }
    .metric-card { background: #f0f2f6; padding: 20px; border-radius: 10px; }
    .prediction-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                      color: white; padding: 20px; border-radius: 15px; font-size: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ─── 1. Load Model & Data ───
@st.cache_resource
def load_model():
    return joblib.load('car_price_MLR_model.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('Cleaned car data.csv')

model = load_model()
data = load_data()

# ─── 2. Sidebar ───
st.sidebar.image("https://img.icons8.com/color/96/car--v1.png", width=80)
st.sidebar.title("🚗 Car Price Predictor")
st.sidebar.markdown("---")

st.sidebar.header("Enter Car Details")

# Categorical
name = st.sidebar.selectbox("Brand / Name", sorted(data['name'].unique()))
fuel = st.sidebar.selectbox("Fuel Type", sorted(data['fuel'].unique()))
transmission = st.sidebar.selectbox("Transmission", sorted(data['transmission'].unique()))
owner = st.sidebar.selectbox("Owner", sorted(data['owner'].unique()))
seller_type = st.sidebar.selectbox("Seller Type", sorted(data['seller_type'].unique()))

st.sidebar.markdown("---")
st.sidebar.header("Technical Specs")

# Numerical
col1, col2 = st.sidebar.columns(2)
with col1:
    car_age = st.number_input("Car Age (yrs)", min_value=0, max_value=30, value=2)
    km_driven = st.number_input("KM Driven", min_value=0, value=10000, step=1000)
    engine = st.number_input("Engine (cc)", min_value=0, value=1248)
with col2:
    max_power = st.number_input("Max Power (bhp)", min_value=0, value=74)
    seats = st.number_input("Seats", min_value=2, max_value=10, value=5)
    mileage = st.number_input("Mileage (km/l)", min_value=0.0, value=23.40, step=0.1)

# ── 3. Main Content with Tabs ───
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Analytics", "ℹ️ About"])

# ─── TAB 1: Prediction ───
with tab1:
    st.markdown('<p class="main-header">Used Car Price Prediction</p>', unsafe_allow_html=True)
    st.write("This app predicts the selling price of a used car based on its features.")

    if st.button("🔍 Predict Price", use_container_width=True, type="primary"):
        input_data = pd.DataFrame({
            'name': [name],
            'km_driven': [km_driven],
            'fuel': [fuel],
            'seller_type': [seller_type],
            'transmission': [transmission],
            'owner': [owner],
            'mileage(km/ltr/kg)': [mileage],
            'engine': [engine],
            'max_power': [max_power],
            'seats': [seats],
            'car_age': [car_age]
        })

        prediction = model.predict(input_data)[0]

        # Show prediction with styling
        st.markdown(f"""
        <div class="prediction-box">
            💰 Predicted Price: <b>₹{prediction:,.2f}</b>
        </div>
        """, unsafe_allow_html=True)

        # Market comparison
        avg_price = data['selling_price'].mean()
        diff = ((prediction - avg_price) / avg_price) * 100
        emoji = "📈" if diff > 0 else ""
        st.metric("vs Market Average", f"₹{avg_price:,.0f}", f"{diff:+.1f}%")

        # Save to session history
        if 'history' not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({
            'Name': name, 'Age': car_age, 'KM': km_driven,
            'Predicted Price': prediction, 'Time': datetime.now().strftime("%H:%M")
        })

        # Show history
        if len(st.session_state.history) > 1:
            st.subheader(" Prediction History")
            st.dataframe(pd.DataFrame(st.session_state.history[-10:]), use_container_width=True)

# ─── TAB 2: Analytics ───
with tab2:
    st.subheader("📊 Market Analytics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Listings", len(data))
    col2.metric("Avg Price", f"₹{data['selling_price'].mean():,.0f}")
    col3.metric("Avg KM Driven", f"{data['km_driven'].mean():,.0f}")

    st.markdown("---")

    # Price by Brand
    st.subheader("Average Price by Brand")
    brand_price = data.groupby('name')['selling_price'].mean().sort_values(ascending=False).head(10)
    fig1 = px.bar(brand_price, x=brand_price.index, y=brand_price.values,
                  color=brand_price.values, color_continuous_scale='Blues',
                  labels={'x': 'Brand', 'y': 'Avg Price (₹)'})
    st.plotly_chart(fig1, use_container_width=True)

    # Price vs KM Driven
    st.subheader("Price vs KM Driven")
    fig2 = px.scatter(data, x='km_driven', y='selling_price', color='fuel',
                      labels={'km_driven': 'KM Driven', 'selling_price': 'Price (₹)'})
    st.plotly_chart(fig2, use_container_width=True)

    # Fuel type distribution
    st.subheader("Fuel Type Distribution")
    fig3 = px.pie(data, names='fuel', hole=0.4)
    st.plotly_chart(fig3, use_container_width=True)

# ─── TAB 3: About ───
with tab3:
    st.subheader("About This App")
    st.write("""
    This app uses a *Multiple Linear Regression* model trained on used car data
    to predict selling prices based on key features.

    *Features used:*
    - Brand, Fuel Type, Transmission
    - Owner type, Seller type
    - Car Age, KM Driven, Engine, Max Power, Seats, Mileage.
    
    *Regression Metrices:*
    - R2 Score : ~83.13%
    - RMSE:  137742.55
    - MAE:  104452.05
    """)
    st.info("Built with Streamlit + Scikit-Learn 🐍")