import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Weather Prediction App")

data = {
    "Day": [1,2,3,4,5,6,7,8,9,10],
    "Temperature": [30,31,32,33,34,35,36,35,37,38]
}

df = pd.DataFrame(data)

st.write(df)

X = df[["Day"]]
y = df["Temperature"]

model = LinearRegression()
model.fit(X, y)

day = st.number_input("Enter Future Day", min_value=1, value=11)

if st.button("Predict"):
    prediction = model.predict([[day]])[0]
    st.success(f"Predicted Temperature: {prediction:.2f} °C")

st.line_chart(df.set_index("Day"))
