# app.py
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import joblib

# Header
st.title("Giao diện Train Model Machine Learning")
st.write("Tải dữ liệu, chọn thuật toán, và train model.")

# Step 1: Upload file
uploaded_file = st.file_uploader("Tải file CSV dữ liệu", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Dữ liệu ban đầu:")
    st.write(data.head())

    # Select Features and Target
    features = st.multiselect("Chọn các cột Feature", options=data.columns)
    target = st.selectbox("Chọn cột Target", options=data.columns)

    if features and target:
        X = data[features]
        y = data[target]

        # Step 2: Split data
        test_size = st.slider("Chọn tỷ lệ Test size", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        st.write("Dữ liệu đã chia thành công!")

        # Step 3: Train Model
        st.subheader("Chọn thuật toán để Train")
        algorithm = st.selectbox("Chọn thuật toán", ["Random Forest", "Logistic Regression"])

        if st.button("Train Model"):
            if algorithm == "Random Forest":
                model = RandomForestClassifier()
            else:
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            st.write(f"Accuracy của {algorithm}: {acc * 100:.2f}%")

            # Save Model
            model_filename = "trained_model.pkl"
            joblib.dump(model, model_filename)
            st.success(f"Model đã lưu thành công: {model_filename}")
