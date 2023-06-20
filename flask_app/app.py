from flask import Flask, render_template, request, Markup, jsonify
import numpy as np
import pandas as pd
import requests
import pickle
import io


# Loading crop recommendation model

crop_recommendation_model_path = '../models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))

app = Flask(__name__)


# Weather Fetch API

def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = "9d7cde1f6d07ec55650544be1631307e"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


# CROP Prediction from sensor-data

@ app.route('/crop-predict-sensor', methods=['GET'])
def crop_prediction_sensor():
    if request.method == 'GET':
        N = 120
        P = 80
        K = 50
        ph = 7
        sensor_data_url = "https://cropsense-sensor.onrender.com/"
        response = requests.get(sensor_data_url)
        json_response = response.json()

        temperature = json_response["soilTemperature"]
        humidity = json_response["moisture"]
        rainfall = json_response["weatherHumidity"]

        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        print(data)
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]

        crop_prediction_data_to_be_returned = {'prediction': final_prediction}
        return jsonify(crop_prediction_data_to_be_returned)


# Crop Prediction API

@ app.route('/crop-predict', methods=['GET'])
def crop_prediction():
    title = 'CropSense - Crop Recommendation'

    if request.method == 'GET':
        N = int(request.args.get('N'))
        P = int(request.args.get('P'))
        K = int(request.args.get('K'))
        ph = float(request.args.get('ph'))
        rainfall = float(request.args.get('rainfall'))

        # state = request.form.get("stt")
        city = request.args.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            print(data)
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            crop_prediction_data_to_be_returned = {'prediction': final_prediction}
            return jsonify(crop_prediction_data_to_be_returned)

        else:

            return render_template('try_again.html', title=title)


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False)