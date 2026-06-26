import streamlit as st
import pandas as pd
pip install scikit-learn
from sklearn.linear_model import LinearRegression

# ---------------------------
# Title
# ---------------------------
st.title("🌤 Weather Prediction App")
st.write("Predict the temperature based on historical data using Linear Regression.")

# ---------------------------
# Sample Dataset
# ---------------------------
data = {
    "Day": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Temperature": [30, 31, 32, 33, 34, 35, 36, 35, 37, 38]
}

df = pd.DataFrame(data)

# ---------------------------
# Display Dataset
# ---------------------------
st.subheader("Historical Temperature Data")
st.dataframe(df)

# ---------------------------
# Train Linear Regression Model
# ---------------------------
X = df[["Day"]]
y = df["Temperature"]

model = LinearRegression()
model.fit(X, y)

# ---------------------------
# User Input
# ---------------------------
st.subheader("Predict Future Temperature")

future_day = st.number_input(
    "Enter Future Day",
    min_value=1,
    value=11,
    step=1
)

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Temperature"):

    future_df = pd.DataFrame({"Day": [future_day]})
    prediction = model.predict(future_df)[0]

    st.success(
        f"Predicted Temperature on Day {future_day}: **{prediction:.2f} °C**"
    )

# ---------------------------
# Temperature Trend Chart
# ---------------------------
st.subheader("Temperature Trend")

chart_df = df.set_index("Day")
st.line_chart(chart_df)

# ---------------------------
# Show Regression Equation
# ---------------------------
st.subheader("Model Details")

st.write(f"Slope (Coefficient): **{model.coef_[0]:.4f}**")
st.write(f"Intercept: **{model.intercept_:.4f}**")
