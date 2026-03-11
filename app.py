import streamlit as st
import pandas as pd
import joblib

st.title("Hotel Adaptive Pricing System")

# Load trained model
model = joblib.load("models/pricing_model.pkl")

st.header("Booking Information")

# User Inputs
hotel = st.selectbox(
    "Hotel Type",
    ["Resort Hotel", "City Hotel"]
)

lead_time = st.slider(
    "Lead Time (Days)",
    0, 365, 30
)

month = st.selectbox(
    "Arrival Month",
    [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ]
)

customer_type = st.selectbox(
    "Customer Type",
    ["Transient", "Contract", "Transient-Party", "Group"]
)

room_type = st.selectbox(
    "Room Type",
    ["A","B","C","D","E","F","G"]
)

previous_bookings = st.number_input(
    "Previous Successful Bookings",
    0, 50, 0
)

# Convert month to number
month_map = {
"January":1,"February":2,"March":3,"April":4,
"May":5,"June":6,"July":7,"August":8,
"September":9,"October":10,"November":11,"December":12
}

arrival_month = month_map[month]

# Predict Button
if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "hotel":[hotel],
        "lead_time":[lead_time],
        "arrival_month":[arrival_month],
        "reserved_room_type":[room_type],
        "customer_type":[customer_type],
        "previous_bookings_not_canceled":[previous_bookings]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"Recommended Room Price: ₹ {round(prediction,2)}")
