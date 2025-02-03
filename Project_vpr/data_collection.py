import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


openweather_api_key = '186127626c87a52b4d5284bf8b4ce744'
tomtom_api_key = 'uJNV62qhStvRlqcKKutZQv3KfUPg2SKE'


cities = {
    'London': {'lat': 51.5074, 'lon': -0.1278},
    'New York': {'lat': 40.7128, 'lon': -74.0060},
    'Tokyo': {'lat': 35.6895, 'lon': 139.6917},
    'Mumbai': {'lat': 19.0760, 'lon': 72.8777},
    'Sydney': {'lat': -33.8688, 'lon': 151.2093}
}


start_date = datetime.now() - timedelta(days=30)
end_date = datetime.now()
date_range = pd.date_range(start_date, end_date)


weather_data = []
traffic_data = []


for date in date_range:
    for city, coords in cities.items():
        weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={coords['lat']}&lon={coords['lon']}&appid={openweather_api_key}"
        weather_response = requests.get(weather_url).json()
        if 'main' in weather_response and 'weather' in weather_response:
            weather_data.append({
                'city': city,
                'date': date,
                'temp': weather_response['main']['temp'],
                'feels_like': weather_response['main']['feels_like'],
                'weather_main': weather_response['weather'][0]['main'],
                'weather_description': weather_response['weather'][0]['description']
            })


for city, coords in cities.items():
    traffic_url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={coords['lat']},{coords['lon']}&key={tomtom_api_key}"
    traffic_response = requests.get(traffic_url).json()
    if 'flowSegmentData' in traffic_response:
        traffic_data.append({
            'city': city,
            'freeFlowSpeed': traffic_response['flowSegmentData']['freeFlowSpeed'],
            'currentSpeed': traffic_response['flowSegmentData']['currentSpeed']
        })
    else:
        traffic_data.append({
            'city': city,
            'freeFlowSpeed': np.nan,
            'currentSpeed': np.nan
        })


weather_df = pd.DataFrame(weather_data)
traffic_df = pd.DataFrame(traffic_data)


merged_df = pd.merge(weather_df, traffic_df, on='city')


merged_df.to_csv('merged_data.csv', index=False)

print("Data collection completed successfully.")
