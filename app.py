import streamlit as st
import pandas as pd
import numpy as np
from model.prophet_model import load_and_preprocess_data,preprocess_data_for_prophet , forecast_prophet, plot_forecast_results
from pandas.api.types import is_datetime64_any_dtype
import streamlit as st
from PIL import Image

image = Image.open("images/sales.jpg")
image = image.resize((100, 100))  # adjust size as needed

# Create two columns: one for image, one for title
col1, col2 = st.columns([1,8 ])  # Adjust ratio to your liking

with col1:
    st.image(image)

with col2:
    st.title("SALES FORECASTING ")
st.header('Insert the csv file for prediction')
file=st.file_uploader("Upload Sales Dataset",type='csv')
if file is not None:
    
    # Load and preprocess the data ie convert string date into datetime obj
    data = load_and_preprocess_data(file)
    st.header("Content of data")
    st.write(data.head())

    # Column selection for Prophet
    st.subheader("🛠 Select Columns for Forecasting Using Prophet")
    columns = data.columns.tolist()

    ds_column = st.selectbox("Select the Date Column (ds)", options=["None"] + columns)
    y_column = st.selectbox("Select the Target Column (y)", options=["None"] +columns)
    if ds_column != "None" and y_column != "None" and ds_column != y_column:
        if not is_datetime64_any_dtype(data[ds_column]):
            try:
                data[ds_column] = pd.to_datetime(data[ds_column])
                st.success(f"✅ Successfully converted '{ds_column}' to datetime.")
            except Exception as e:
                st.error(f"❌ Error converting '{ds_column}' to datetime: {e}")
                st.stop()  # Stop execution if conversion fails
        else:
            st.success(f"✅ '{ds_column}' is already in datetime format.")

        # Prepare selected data
        df = data[[ds_column, y_column]].copy()
        df = preprocess_data_for_prophet(df, ds_column, y_column)

        st.subheader("✅ Structured Data for Prophet (ds, y)")
        st.write(df.head())

        # Forecast using Prophet
        forecast, mae, rmse, r2, model, test_data = forecast_prophet(df)

        st.subheader("📋Forecasted Results")
        st.write(forecast.head())
        csv = forecast.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Forecast Data", csv, "forecast.csv", "text/csv")

        # Display Metrics
        st.write("### 🧮 Model Accuracy Metrics")
        st.write(f"✔ Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"✔ Root Mean Squared Error (RMSE): {rmse:.2f}")
        st.write(f"✔ R² Score(Higher is better):{r2:.2f} ")

        # Plot Results
        plot_forecast_results(forecast, model, test_data)
    else:
        st.warning("⚠️ Please select two **different** columns for date and target.")

else:
    st.info("📂 Please upload a CSV file to begin.")

