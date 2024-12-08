import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, jsonify
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from io import BytesIO
from datetime import datetime
import traceback

app = Flask(__name__)

# MongoDB connection details
MONGO_URI = "mongodb+srv://kumarpriyanshu762:aquameter@cluster0.ez27m.mongodb.net/Aquameter"
DATABASE_NAME = "Aquameter"
COLLECTION_NAME = "dailyaveragewqis"  # Collection where you store your actual data
FORECAST_COLLECTION = "forecasted_wqis"  # Collection to store the forecasted values

# Connecting to MongoDB
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]
forecast_collection = db[FORECAST_COLLECTION]  # Forecast collection

# Load or train ARIMA model per device
def load_or_train_arima_model(data):
    try:
        # Fit ARIMA model for the provided data
        model = ARIMA(data['WQI'], order=(1, 1, 1))  # Adjust the order to your needs
        model_fit = model.fit()
        return model_fit
    except Exception as e:
        print(f"Error while training the ARIMA model: {e}")
        return None

# API route to fetch data and provide a 7-day forecast
@app.route('/', methods=['GET'])
def api_forecast():
    try:
        # Fetch all unique deviceIds
        device_ids = collection.distinct("deviceId")
        if not device_ids:
            return jsonify({"error": "No devices found in the database"}), 404
        
        # Prepare a dictionary to hold forecast data for each device
        all_forecast_data = {}

        # Iterate over each deviceId and generate forecast for the latest 7 days of data
        for device_id in device_ids:
            # Fetch the latest 7 records for this device
            records = list(collection.find({"deviceId": device_id}).sort('Date', -1).limit(7))
            if not records:
                continue  # Skip if no data found for this device

            # Convert MongoDB records to DataFrame
            data = pd.DataFrame(records)

            # Ensure 'WQI' and 'Date' columns exist
            if 'WQI' not in data.columns or 'Date' not in data.columns:
                continue  # Skip if data is missing required columns

            # Parse 'Date' column as datetime
            data['Date'] = pd.to_datetime(data['Date'])

            # Sort data by Date (if needed)
            data = data.sort_values(by='Date')

            # Forecast with the ARIMA model for this device
            arima_model = load_or_train_arima_model(data)
            if arima_model is None:
                continue  # Skip if ARIMA model training failed for this device

            # Forecast the next 7 days
            forecast_steps = 7  # Forecast next 7 steps
            forecast_values = arima_model.forecast(steps=forecast_steps)

            # Prepare the forecasted data (dates)
            forecast_dates = pd.date_range(data['Date'].max(), periods=forecast_steps + 1, freq='D')[1:]
            forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast_WQI': forecast_values})

            # Update the forecast collection with the forecasted data for this deviceId
            existing_forecast = forecast_collection.find_one({"deviceId": device_id})

            if existing_forecast:
                # If a document exists for the deviceId, overwrite the forecasted data
                forecast_collection.update_one(
                    {"deviceId": device_id},
                    {"$set": {"forecasted_data": [{"Date": row['Date'].strftime('%a, %d %b %Y %H:%M:%S GMT'), "Forecast_WQI": row['Forecast_WQI']} for index, row in forecast_df.iterrows()]}}
                )
            else:
                # If no document exists for the deviceId, create a new one with the forecasted data
                forecast_data = [{"Date": row['Date'].strftime('%a, %d %b %Y %H:%M:%S GMT'), "Forecast_WQI": row['Forecast_WQI']} for index, row in forecast_df.iterrows()]
                forecast_collection.insert_one({
                    "deviceId": device_id,
                    "forecasted_data": forecast_data
                })

            # Store the forecast data for the device in the dictionary
            all_forecast_data[device_id] = forecast_df.to_dict(orient='records')

        # Print the forecasted data to the terminal for each device
        print("Forecasted Data (DeviceId, Date, Forecast_WQI):")
        for device_id, forecast_data in all_forecast_data.items():
            print(f"Device {device_id}:")
            print(forecast_data)

        # Plot the actual and forecasted data for the first device as an example
        first_device_data = pd.DataFrame(list(collection.find({"deviceId": device_ids[0]}).sort('Date', -1).limit(7)))
        first_device_data['Date'] = pd.to_datetime(first_device_data['Date'])

        plt.figure(figsize=(10, 6))
        plt.plot(first_device_data['Date'], first_device_data['WQI'], label='Actual WQI', color='blue')
        plt.plot(forecast_df['Date'], forecast_df['Forecast_WQI'], label='Forecast WQI', color='orange')
        plt.title(f'WQI Forecast for the Next 7 Days (Device {device_ids[0]})')
        plt.xlabel('Date')
        plt.ylabel('WQI')
        plt.legend()
        plt.grid()

        # Save the plot to a BytesIO object
        plot_img = BytesIO()
        plt.savefig(plot_img, format='png')
        plot_img.seek(0)
        plt.close()

        # Save the plot to static folder for access
        plot_path = "static/forecast_plot.png"
        with open(plot_path, "wb") as f:
            f.write(plot_img.getbuffer())

        # Return the forecasted values and path to the plot
        return jsonify({
            "forecast": all_forecast_data,
            "plot_path": plot_path
        })

    except Exception as e:
        # Log the error details to Flask console and return detailed message to the user
        error_details = str(e)
        traceback_details = traceback.format_exc()

        print(f"Error: {error_details}")
        print(f"Stack Trace: {traceback_details}")

        return jsonify({"error": f"Internal Server Error: {error_details}"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
