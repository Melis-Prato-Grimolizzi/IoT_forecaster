import configparser
import time
import matplotlib.pyplot as plt
import SessionHTTP as Http
from prophet import Prophet
import pandas as pd
from datetime import datetime
import json


class Forecaster:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.bearer = None
        self.model = None
        self.data = pd.DataFrame()
        self.datas = {}
        self.forecasted_data = pd.DataFrame()
        self.base_url = self.config.get('Urls', 'BaseUrl')

    def forecasterLogin(self):
        session = Http.getSession()
        username = self.config['Account']['username']
        password = self.config['Account']['password']
        login_url = self.config['Urls']['Login']
        url = self.base_url + login_url
        data = {
            'username': username,
            'password': password
        }
        response = session.post(url, data=data)
        self.bearer = 'Bearer ' + response.text
        print(f"DEBUG: Response for server (Login): {response.text}")

    def     getHistoricalData(self, periods=12):
        session = Http.getSession()
        url = self.config['Urls']['GetHistory']
        url = self.base_url + url + str(periods)
        print(f"DEBUG: URL for historical data: {url}")
        header = {
            'Authorization': self.bearer
        }
        response = session.get(url, headers=header)
        self.data = pd.DataFrame(response.json())

        # print(f"Data: {self.data}")
        # print(self.data.info())
        # print(self.data.describe())
        # self.data.to_csv('data.csv')

        #print(f"DEBUG: Response for server (Historical data): {response.text}")

    def transformData(self):
        self.datas = {parking_id: group for parking_id, group in self.data.groupby('parking_id')}
        for df in self.datas:
            self.datas[df]['ds'] = self.datas[df]['timestamp'].apply(
                lambda x: datetime.utcfromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S')
            )
            self.datas[df].drop(columns=['timestamp'], inplace=True)
            self.datas[df].rename(columns={'state': 'y'}, inplace=True)
            self.datas[df] = self.datas[df].sort_values(by=['ds']).reset_index(drop=True)
            self.datas[df]['y'] -= self.datas[df]['y'].mean()

    def transformNewData(self):
        new = {parking_id: group for parking_id, group in self.data.groupby('parking_id')}
        for df in new:
            new[df]['ds'] = new[df]['timestamp'].apply(
                lambda x: datetime.utcfromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S')
            )
            new[df].drop(columns=['timestamp'], inplace=True)
            new[df].rename(columns={'state': 'y'}, inplace=True)
            new[df] = new[df].sort_values(by=['ds']).reset_index(drop=True)
            new[df]['y'] -= new[df]['y'].mean()
        return new

    def updateData(self, new_data, periods=12):
        for df in new_data:
            self.datas[df] = self.datas[df].iloc[periods:]
            self.datas[df] = pd.concat([self.datas[df], new_data[df]]).reset_index(drop=True)

        print('Data updated!')

    def postForecast(self):
        session = Http.getSession()
        url = self.config['Urls']['PostForecast']
        url = self.base_url + url
        header = {
            'Authorization': self.bearer,
            'Content-Type': 'application/json'
        }
        json_data = json.dumps(self.forecasted_data.values.tolist())
        print(f"DEBUG: Data to post: {json_data}")

        # print(json_data)
        response = session.post(url, headers=header, data=json_data)
        print(f"DEBUG: Response for server (Post forecast): {response.text}")

    def fit_predict(self, periods=12):
        for slot_history in self.datas.values():
            self.model = Prophet(growth='linear', changepoint_prior_scale=0.7)
            self.model.add_seasonality(name='hourly', period=24, fourier_order=70)
            self.model.add_seasonality(name='minutely', period=6, fourier_order=50)
            parking_id = slot_history['parking_id'].values[0]
            self.model.fit(slot_history)
            new_forecast = self.forecast(parking_id, periods=12)
            self.forecasted_data = pd.concat([self.forecasted_data, new_forecast])

    def forecast(self, parking_id, periods=12):
        future = self.model.make_future_dataframe(periods=144,
                                                  freq='10min',
                                                  include_history=False)
        forecast = self.model.predict(future)
        print('Forecasting done')
        predicted_mean = forecast['yhat'].mean()
        forecast['yhat_bin'] = (forecast['yhat'] > predicted_mean - 0.06).astype(int)
        forecast['parking_id'] = parking_id
        forecast['ds'] = forecast['ds'].apply(
            lambda x: datetime.timestamp(x)
        )
        ret_forecast = pd.DataFrame()
        ret_forecast['ds'] = forecast['ds']
        ret_forecast['parking_id'] = forecast['parking_id']
        ret_forecast['yhat_bin'] = forecast['yhat_bin']
        # print(ret_forecast)
        # plt.plot(ret_forecast['ds'], ret_forecast['yhat_bin'], label='Previsioni')
        # plt.title('Predictions')
        # plt.show()
        return ret_forecast.iloc[0:periods]

    def loop(self):
        timestamp = time.time()
        first = True
        try:
            while True:
                if (time.time() - timestamp >= 7200) or first:
                    first = False
                    timestamp = time.time()
                    if self.data.shape[0] == 0:
                        self.getHistoricalData(3600)
                        print("Historical data collected!")
                        self.transformData()
                        self.fit_predict(periods=12)
                        self.postForecast()
                    else:
                        self.getHistoricalData(12)
                        new_data = self.transformNewData()
                        self.updateData(new_data, periods=12)
                        self.fit_predict(periods=12)
                        self.postForecast()
                        break
                time.sleep(1)
        except KeyboardInterrupt:
            print("Forecaster exiting...")
            time.sleep(2)


if __name__ == '__main__':
    f = Forecaster()
    f.forecasterLogin()
    f.loop()
