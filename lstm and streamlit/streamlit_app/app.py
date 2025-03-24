import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

FLASK_URI= "UPDATE_URL_HERE"

@st.cache_data()
def fetch_data():
    response= requests.get(FLASK_URI)
    if response.status_code==200:
        data= response.json()

        df_=pd.DataFrame({
            "date": data["dates"],
            "temperature": data["past_7days_temperature"]
        })

        df_["date"]= pd.to_datetime(df_["date"])
        df_.set_index("date", inplace=True)

        predicted_temp= data["prediction_temperature"]
        
        return df_, predicted_temp
    
    else:
        return None, None
    
st.title("Weather Forecast Prediction")

df, predicted_temp= fetch_data()

if df is not None:
    # Plot last 7 days' temperatures
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=df.index, y=df["temperature"], marker="o", ax=ax, label="Last 7 Days")

    # Mark the predicted temperature
    predicted_date = datetime.strptime(datetime.now().strftime("%Y-%m-%d"), "%Y-%m-%d") + timedelta(days=1)

    last_date = df.index[0].date()
    last_temp = df["temperature"].iloc[0]

    # Add a dotted line from last temperature to predicted temperature
    ax.plot([last_date, predicted_date], [last_temp, predicted_temp], linestyle="dotted", color="red", label="Prediction Trend")


    ax.scatter(predicted_date, predicted_temp, color="red", s=50, label="Predicted (Today)")
    ax.text(predicted_date, predicted_temp, f"{predicted_temp:.1f}°C", fontsize=12, ha="right", color="green")

    # Formatting
    ax.set_title("Temperature Trend (Last 7 Days & Prediction)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    plt.xticks(rotation=30)

    # Display plot in Streamlit
    st.pyplot(fig)

    # Show predicted temperature
    st.markdown(f"### 🔮 Predicted Temperature for **Tomorrow ({predicted_date.date()})**: `{predicted_temp:.1f}°C`")
else:
    st.error("Failed to fetch data. Please check API connection.")

