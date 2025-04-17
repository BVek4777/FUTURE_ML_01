import streamlit as st
import pandas as pd
import numpy as np
from model.prophet_model import load_and_preprocess_data,preprocess_data_for_prophet , forecast_prophet, plot_forecast_results


st.title("SALES FORECASTING FOR RETAIL BUSINESS")
st.header('Insert the csv file for prediction')
file=st.file_uploader("Upload Sales Dataset",type='csv')
if file is not None:
    
    # Load and preprocess the data
    data = load_and_preprocess_data(file)
    st.header("Content of data")
    st.write(data.head())
    
    # Prepare data for Prophet model
    df = data[['ORDERDATE', 'SALES']]
    df = preprocess_data_for_prophet(df)
    
    # Show the structured data
    st.header("Structuring of data into two columns ds & y")
    st.write(df)
    
    # Forecast using Prophet
    forecast, mae, rmse, r2,model,test_data = forecast_prophet(df)
     # Display forecasted data
    st.header("Forecasted Data")
    st.write(forecast[:5])
    csv = forecast.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Forecast Data", csv, "forecast.csv", "text/csv")

   

    # Display Accuracy Results
    st.write("### 📊 Model Accuracy Metrics")
    st.write(f"✔ Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"✔ Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"✔ R² Score: {r2:.2f} (Higher is better)")

     # Plot Actual vs Predicted Sales
    plot_forecast_results(forecast,model,test_data)

else:
     st.warning("⚠️ Please upload a CSV file to proceed with forecasting.")  
   
    