import streamlit as st
import requests

# URLs for Flask APIs
TRAJECTORY_API = "http://127.0.0.1:5000/predict_trajectory"
OBJECT_API = "http://127.0.0.1:5000/predict_object"

# Streamlit app title and sidebar
st.title("Near-Earth Object (NEO) Prediction")

# Section for Trajectory Prediction
st.sidebar.header("Input Features for Trajectory Prediction")

# User input for 4 timesteps (adjusting the number of timesteps to match your model's expected input)
timesteps = []
for i in range(1, 5):
    features = [
        st.sidebar.number_input(f"Timestep {i} - Relative Velocity (km/s)", value=25.0),
        st.sidebar.number_input(f"Timestep {i} - Miss Distance (km)", value=1e6),
        st.sidebar.number_input(f"Timestep {i} - Min Diameter (km)", value=0.1),
        st.sidebar.number_input(f"Timestep {i} - Max Diameter (km)", value=0.5),
    ]
    timesteps.append(features)

# Predict trajectory button
if st.button("Predict Trajectory"):
    with st.spinner("Sending data to Flask API..."):
        try:
            # Send the input data to the API
            response = requests.post(TRAJECTORY_API, json={"features": timesteps})
            
            if response.status_code == 200:
                trajectory = response.json()['trajectory']
                st.success(f"Predicted Trajectory (Miss Distance): {trajectory}")
            else:
                st.error("Error in Trajectory Prediction API")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection Error: {e}")

# Section for Object Identification
st.sidebar.header("Input Features for Object Identification")

# User input for object identification
feature_1 = st.sidebar.number_input("Feature 1 (Absolute Magnitude)", value=0.5)
feature_2 = st.sidebar.number_input("Feature 2 (Relative Velocity)", value=0.2)
feature_3 = st.sidebar.number_input("Feature 3 (Min Diameter)", value=1.1)

# Collect all input features
input_features = [feature_1, feature_2, feature_3]

# Identify object button
if st.button("Identify Object"):
    with st.spinner("Sending data to Flask API..."):
        try:
            # Send input data to the API
            response = requests.post(OBJECT_API, json={"features": input_features})

            if response.status_code == 200:
                result = response.json()
                st.success(f"Predicted Object Type: {result['object_type']}")
            else:
                error_message = response.json().get('error', 'Unknown Error')
                st.error(f"API Error: {error_message}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection Error: {e}")
