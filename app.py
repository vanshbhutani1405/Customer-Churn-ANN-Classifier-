import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

#Loading
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_geo.pkl', 'rb') as file:
    one_hot_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered"
)


st.markdown("<h1 style='text-align: center;'>📉 Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Predict whether a customer is likely to leave using a Deep Learning ANN model</p>",
    unsafe_allow_html=True
)

st.divider()


st.sidebar.header("🔧 Enter Customer Details")

geography = st.sidebar.selectbox('Geography', one_hot_geo.categories_[0])
gender = st.sidebar.selectbox('Gender', label_encoder_gender.classes_)
age = st.sidebar.slider('Age', 18, 92, 30)
balance = st.sidebar.number_input('Balance', value=50000.0)
credit_score = st.sidebar.number_input('Credit Score', value=600)
estimated_salary = st.sidebar.number_input('Estimated Salary', value=50000.0)
tenure = st.sidebar.slider('Tenure (years)', 0, 10, 3)
num_of_products = st.sidebar.slider('Number of Products', 1, 4, 1)
has_cr_card = st.sidebar.selectbox('Has Credit Card', [0, 1])
is_active_member = st.sidebar.selectbox('Is Active Member', [0, 1])


st.subheader("📋 Customer Snapshot")
col1, col2 = st.columns(2)

with col1:
    st.write(f"**Geography:** {geography}")
    st.write(f"**Gender:** {gender}")
    st.write(f"**Age:** {age}")
    st.write(f"**Tenure:** {tenure}")
    st.write(f"**Credit Score:** {credit_score}")

with col2:
    st.write(f"**Balance:** {balance}")
    st.write(f"**Products:** {num_of_products}")
    st.write(f"**Has Credit Card:** {has_cr_card}")
    st.write(f"**Active Member:** {is_active_member}")
    st.write(f"**Salary:** {estimated_salary}")

st.divider()


if st.button("🚀 Predict Churn", use_container_width=True):

    # Prepare input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode Geography
    geo_encoded = one_hot_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=one_hot_geo.get_feature_names_out(['Geography'])
    )

    # Combine all features
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.divider()
    st.subheader("📊 Prediction Result")

    st.metric("Churn Probability", f"{prediction_proba:.2%}")

    if prediction_proba > 0.5:
        st.error("⚠️ The customer is likely to churn.")
    else:
        st.success("✅ The customer is likely to stay.")


st.divider()
st.markdown(
    "<p style='text-align: center; color: gray;'>Built using ANN & Streamlit</p>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: gray;'>Made by Vansh</p>",
    unsafe_allow_html=True
)
