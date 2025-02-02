import configparser
import SessionHTTP as Http
import prophet as Prophet


class Forecaster:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.bearer = None
        self.model = None
        self.data = None
        self.forecasted_data = None

    def forecasterLogin(self):
        session = Http.getSession()
        username = self.config['Account']['username']
        password = self.config['Account']['password']
        login_url = self.config['Urls']['Login']
        data = {
            'username': username,
            'password': password
        }
        response = session.post(login_url, data=data)
        self.bearer = 'Bearer ' + response.text
        print(f"DEBUG: Response for server (Login): {response.text}")

    def getHistoricalData(self):
        session = Http.getSession()
        url = self.config['Urls']['GetHistory']
        header = {
            'Authorization': self.bearer
        }
        response = session.get(url, headers=header)
        # TODO: save data in self.data
        print(f"DEBUG: Response for server (Historical data): {response.text}")

    def postForecast(self):
        session = Http.getSession()
        url = self.config['Urls']['PostForecast']
        header = {
            'Authorization': self.bearer
        }
        data = {
            # TODO: capire bene che dati mandare e in che formato
            'forecast': self.forecasted_data
        }
        response = session.post(url, headers=header, data=data)
        print(f"DEBUG: Response for server (Post forecast): {response.text}")

    def fit(self, df_train):
        self.model = Prophet(growth='linear', change_point_prior_scale=0.7)
        self.model.add_seasonality(name='hourly', period=24, fourier_order=70)
        self.model.add_seasonality(name='minutely', period=6, fourier_order=50)
        self.model.fit(df_train)

    def forecast(self, periods):
        future = self.model.make_future_dataframe(periods=periods,
                                                  freq='10min',
                                                  include_history=False)
        forecast = self.model.predict(future)
        print('Forecasting done')
        predicted_mean = forecast['yhat'].mean()
        forecast['yhat_bin'] = (forecast['yhat'] > predicted_mean - 0.06).astype(int)
        self.forecasted_data = forecast

    def loop(self):
        while True:
            self.getHistoricalData()
            self.fit(df_train=self.data)
            self.forecast(periods=144)
            self.postForecast()


if __name__ == '__main__':
    f = Forecaster()
    f.forecasterLogin()
    f.loop()
