# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly


# Create time series forecasting function using prophet
def run_forecast(df, fcst_length, include_history = False):

  # Create forecast and fit model
  forecast_dimensions = df.drop(columns = ['ds', 'y']).drop_duplicates()

  # Loop through each dimension and forecast, placing into a data frame
  output = pd.DataFrame()

  for i in range(0, forecast_dimensions.shape[0]):

    # Prep historicals
    join_dims = forecast_dimensions.iloc[i:i+1,:]
    join_keys = list(join_dims.columns)
    historicals = pd.merge(df, join_dims, on = join_keys)

    # Build and fit model
    m = Prophet(daily_seasonality=False)
    m.add_country_holidays(country_name='US')
    m.fit(historicals[['ds', 'y']])

    # Create data frame for forecasts
    future = m.make_future_dataframe(periods=(fcst_length))

    # Forecast and keep date and the predicted value only
    forecast = m.predict(future)
    forecast = forecast[['ds', 'yhat']]

    # join against the unique dims we forecasted across. Create a "key" column in both tables that
    # is set to 1, and join (acts as cross join)
    forecast['key'] = 1
    join_dims['key'] = 1
    finalized_forecast = pd.merge(forecast, join_dims, on = 'key').drop('key', 1)
    finalized_forecast = finalized_forecast.loc[~finalized_forecast['ds'].isin(list(historicals['ds']))]

    # Combine remove historicals and output
    output = pd.concat([output, finalized_forecast], axis = 0)

  # Return output
  if include_history == True:
    output = pd.concat([historicals, output], axis = 0)

  output.loc[:,'ds'] = output['ds'].astype(str)

  return output
  
  
# Forecast 
output = run_forecast(input_data, 365, include_history = False)
