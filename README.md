# IoT Forecaster Documentation

## Introduction
This document provides a brief overview of the IoT Forecaster project. 
The IoT Forecaster is a web application that allows the forecasting of the future parking slot occupancy. 
The application uses a machine learning model to predict the future state of a slot, based on historical data, in particular, Prophet.

## Workflow
This script is organized in a loop that runs every 30 minutes.
The loop works as follows:
1. The script uses the `getHistoricalData` that allows, through an HTTP request, to retrieve a specified number of historical samples (see `samples` variable) from the database.
2. The second part of the loop is the data transoformation, pergormed by the `transformData` function.
This function unpacks the, eventually unordered, data and it creates a list of dictionaries, where each dictionary contains the slot id, the timestamp and the occupancy status.
At this point we apply a transformation to the data, in particular we convert the timestamp to a datetime object and we sort the data by timestamp.
Finally, we subtract the state occupancy mean from the occupancy status, in order to have a zero mean.
3. At this point, if we are not in the first iteration, we must update our saved historical data.
This is perfomed in the `updateData` function, in such a smart way that keep open the possibility to update the data in a
non fixed time window; to do this, we first discard a precise number (`periods`) of samples from the historical data that we own,
then we append the new data to the historical data and we save it.
4. At this point, we perform the `fit_predict` function, that define a specific prophet model for each slot and it fits the model to the historical data.
The model is defined as follows:
```python
self.model = Prophet(growth='linear', changepoint_prior_scale=0.7)
self.model.add_seasonality(name='hourly', period=24, fourier_order=70)
self.model.add_seasonality(name='minutely', period=6, fourier_order=50)
```
Now we can predict the future state of the slot, and concatenate the prediction in order to build the forecast object.
5. Now we use the `postForecast` function to save the forecast in the backend.
This function clearly make a POST request to the backend, sending the forecast object in the body of the request.


## Forecasting 
The forecasting action is performed inside the `fit_predict` function.
We can spend a word on the way the forecasting is discretized;

```python
forecast['yhat_bin'] = (forecast['yhat'] > predicted_mean - 0.06).astype(int)
```

In this way we discretize the forecast in a binary way, where the occupancy status is 1 if the forecasted occupancy is greater than the mean predicted occupancy, and 0 otherwise.
In order to make the forecast possible, we need to make the forecast for an entire day, so we need to forecast 144 samples, each one 10 minutes apart from the previous one.
Even if the forecast that we need is for the next 2 hours, we need to forecast the entire day, because the discretization of the forecast is based on the mean occupancy of the slot, and we need to have the mean occupancy of the slot for the entire day.
