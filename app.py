import streamlit as st
import joblib
import numpy as np

model = joblib.load('house_predict_rf.pkl')
locality_cluster_map = joblib.load("locality_cluster_map.pkl")

st.title("House Price Prediction App")
property_type = st.number_input("BHK (Property Type)", min_value=1, max_value=10, step=1)
area_sqft = st.number_input("Area in Sqft", min_value=100.0)
age_of_property = st.number_input("Age of Property (years)", min_value=0.0)

construction_status = st.selectbox("Construction Status", [
    'Under Construction', 'New', 'Ready to move', 'Resale'
])
localities = sorted(locality_cluster_map.keys())
selected_locality = st.selectbox("Select Locality", localities)

locality_cluster = locality_cluster_map[selected_locality]

status_map = {'Under Construction': 4, 'New': 3, 'Ready to move': 2, 'Resale': 1}
construction_status = status_map[construction_status]

# Prepare the features array for prediction
features = np.array([[property_type, area_sqft, construction_status, age_of_property, locality_cluster]])

# Predict the price
log_price = model.predict(features)[0]
predicted_price = np.expm1(log_price)  # Reverse the log transformation

# Display the predicted price
st.subheader("💰 Predicted Price:")
st.success(f"₹ {predicted_price:.2f} Crores")