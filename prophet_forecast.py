# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import math

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.make_holidays import make_holidays_df, hdays_part1

# Create the data frame for all holidays used in the model
def generate_holidays(df, fcst_length_year=3):
  
  # Grab all supported country codes from prophet and put into one table
  holiday_countries = hdays_part1.list_supported_countries()
  
  # Create empty data frame - when we extract holidays, we union each country with
  # the empty df until we have a complete df
  holidays = pd.DataFrame(
      {
        'holiday': 'test',
        'ds': pd.to_datetime(['2000-01-01', '2000-01-02'])
      }
    )

  # Bring in holidays for each country by looping through each country available
  for i in holiday_countries:
    
    # Grab holidays for holiday i in the loop. Grab holidays from the start of the
    # input data to the end of the input data + some forecast length
    tmp = make_holidays_df(year_list = range(
      df.loc[:,['ds']].min()[0].year, 
      df.loc[:,['ds']].max()[0].year + fcst_length_year, 1), country = i)
    
    # Combine the temp dataset with the complete one
    holidays = pd.concat([holidays, tmp])

  # Get rid of duplicate holidays and add add a window from -3 to +5
  holidays = holidays.drop_duplicates()
  holidays['lower_window'] = -3
  holidays['upper_window'] = 5

  return holidays
  
  
# Create time series forecasting function using prophet
def run_forecast(df, fcst_length, include_history = False):

  # Create forecast and fit model
  forecast_dimensions = df.drop(columns = ['ds', 'y']).drop_duplicates()

  # Create holiday dataset to pass along into prophet
  holiday_list = generate_holidays(df = df, fcst_length_year = math.ceil(fcst_length / 365))
  
  # Loop through each dimension and forecast, placing into a data frame
  output = pd.DataFrame()

  for i in range(0, forecast_dimensions.shape[0]):

    # Prep historicals
    join_dims = forecast_dimensions.iloc[i:i+1,:]
    join_keys = list(join_dims.columns)
    historicals = pd.merge(df, join_dims, on = join_keys)

    # Build and fit model
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, holidays=holiday_list)
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
