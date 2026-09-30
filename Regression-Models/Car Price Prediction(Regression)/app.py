import streamlit as st
import joblib
import pandas as pd

# 1 Load Model using catching for fast performance.
st.cache_resource
def load_model():
    return joblib.load('car_price_MLR_model.pkl')
model = load_model()
data = pd.read_csv('Cleaned car data.csv')

# 2-------------------- App Title ---------------------------------------------------
st.title('Used Car Price Prediction')
st.write('This app Predicts the sellong price of used car based on its features')

#3 -------------------------Sidebar for inputs-----------------------------------------
st.sidebar.header('Enter Car Details')

# ------------------------categorical features-----------------------------------------
name = st.sidebar.selectbox('Name',data['name'].unique())
fuel = st.sidebar.selectbox('Fuel Type',data['fuel'].unique())
transmission = st.sidebar.selectbox('Transmission',data['transmission'].unique())
owner = st.sidebar.selectbox('Owner',data['owner'].unique())
seller_type = st.sidebar.selectbox('Seller Type',data['seller_type'].unique())
# ----------------------------numerical features----------------------------------------
car_age = st.sidebar.number_input('Car Age',min_value=0,value=2)
km_driven = st.sidebar.number_input('KM Driven',min_value=0,value=10000)
engine	= st.sidebar.number_input('Engine',min_value=0,value=1248)
max_power = st.sidebar.number_input('Max Power',min_value=0,value=74)
seats = st.sidebar.number_input('Seats',max_value=7,min_value=2,value=5)
mileage = st.sidebar.number_input('Mileage',value=23.40)


# 4--------------------------- Prediction Button----------------------------------------
if st.button('Predict Price'):
    input_data = pd.DataFrame({
        'name':[name],
        'km_driven':[km_driven],
        'fuel':[fuel],
        'seller_type': [seller_type],
        'transmission':[transmission],
        'owner':[owner],
        'mileage(km/ltr/kg)':[mileage],
        'engine':[engine],
        'max_power':[max_power],
        'seats': [seats],
        'car_age':[car_age]
        
    })
    # --------------------------------prediction------------------------------------------
    prediction = model.predict(input_data)
    #result
    st.success(f"Predicted Price: {prediction[0]:.2f}")