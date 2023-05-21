# Import libraries
import pandas as pd
import numpy as np
import math
import itertools
from datetime import datetime

from prophet import Prophet
from prophet.make_holidays import make_holidays_df, hdays_part1
from prophet.diagnostics import cross_validation, performance_metrics

# Create the data frame for all holidays used in the model
def generate_holidays(df, fcst_length_year=3, countries = []):
  
    """
    Generates a list of holidays by country for a given time range. The function takes a list of countries for which it
    returns all the holidays from. It also takes a data frame of historical values and finds the earliest date 
    in that data frame. It then returns all holidays between the earliest date of the input data and the end of the
    forecast period, which is the final date of the input data plus the forecast length in years.
        
    Args:
        df: A data frame of historical data to be forecasted. Any number of columns can be inputted, but the date column
            must be called "ds" and the metric to forecast must be called "y"
        
        fcst_length_year: The number of years ahead of the final date in the historical data frame to return holidays
            for. For example, if the input data goes from Jan 1, 2010 to Dec 31, 2012, and the fcst_length_year = 3, 
            Then the holiday range goes from 2010 (start date) to 2015 (2012 + 3)

    Returns:
        A data frame of holidays by date. It returns the holiday name, the date of that holiday, the country of that
        holiday, and a lower & upper bound. The lower bound is the number of days before the holiday that we expect
        the impact of the holiday to occur, and the upper bound is the number of days after the holiday that we expect
        the impact to still be felt. For example, downtrend in online activity happens before and after Christmas, so
        we can't only look for an effect on Christmas. Lower defaults to 3 days, upper defaults to 5 
    """
    
    # Grab all supported country codes from prophet and put into one table
    holiday_countries = hdays_part1.list_supported_countries()
  
    # Create empty data frame - when we extract holidays, we union each country with
    # the empty df until we have a complete df
    holidays = pd.DataFrame(columns=['holiday', 'ds'])
  
    # For no inputs, use all countries. Otherwise use select countries
    if len(countries) == 0:
  
        # Bring in holidays for each country by looping through each country available
        for i in holiday_countries:
            
            # Grab holidays for holiday i in the loop. Grab holidays from the start of the
            # input data to the end of the input data + some forecast length
            start_year = datetime.strptime(df.loc[:,['ds']].min()[0], "%Y-%m-%d").year
            end_year = datetime.strptime(df.loc[:,['ds']].max()[0], "%Y-%m-%d").year
            tmp = make_holidays_df(year_list = range(start_year, end_year + fcst_length_year + 1, 1), country = i)
            tmp['country'] = i

            # Combine the temp dataset with the complete one
            holidays = pd.concat([holidays, tmp])
      
        # Use select countries
    elif len(countries) > 0:
        
        # Bring in holidays for each country by looping through each country available
        for i in holiday_countries:
            
            # Only use countries of interest
            if i in countries:
                
                # Grab holidays for holiday i in the loop. Grab holidays from the start of the
                # input data to the end of the input data + some forecast length
                start_year = datetime.strptime(df.loc[:,['ds']].min()[0], "%Y-%m-%d").year
                end_year = datetime.strptime(df.loc[:,['ds']].max()[0], "%Y-%m-%d").year
                tmp = make_holidays_df(year_list = range(start_year, end_year + fcst_length_year, 1), country = i)
                tmp['country'] = i

                # Combine the temp dataset with the complete one
                holidays = pd.concat([holidays, tmp])
    

    # Get rid of duplicate holidays and add add a window from -3 to +5
    holidays = holidays.drop_duplicates()
    holidays['lower_window'] = -3
    holidays['upper_window'] = 5

    return holidays
  
  
# Create time series forecasting function using prophet
def run_forecast(df, 
                 fcst_length, 
                 country_list=['US', 'CN'], 
                 uncertainty_samples_input=0, 
                 perform_cross_validation=False, 
                 cross_validation_param_grid=False):
    """
    Calls the Prophet model to run time-series forecasting. This function provides a wrapper around the regular Prophet
    model, which can only take one input and provide one output. This wrapper will take a df of multiple dimensions and
    run a forecast for each before unioning all the outputs together. In addition, the model can perform model selection
    through cross_validation to determine the best model to run.

    Args:
        df: A data frame of historical data to be forecasted. Any number of columns can be inputted, but the date column
            must be called "ds" and the metric to forecast must be called "y"
        
        fcst_lebgth: The number of days to be forecasted
        
        country_list: The list of countries to pull for holiday effects. The more countries, the longer the forecast will
            take because it needs to cross-check against more days. Defaults to US and China as that covers most major
            holidays
            
        uncertainty_samples_input: The number of samples for calculating uncertainty intervals. If set to 0, will only give
            a point forecast but runs faster. If set to 1000, will give a range around the forecast but will run slower
            
        perform_cross_validation: a True/False input of whether cross-validation is performed. If so, the function will
            run back-testing on each input time-series using a variety of hyperparameters and calculate RMSE for each. It
            will then pick the set of paramaters with the lowest RMSE and use that for the main forecast. If disabled, 
            the model will run much faster but won't do any model selection
            
        cross_validation_param_grid: The grid of parameters used for backtesting. If false, it will set to a grid
            of default values

    Returns:
        A data frame of time series forecasts for each individual set of dimensions in the input_df.
    """
    
    
    # If there are no other columns than date and output, add a temporary column so the 
    # partition below works. Drop the column later
    if df.shape[1] == 2:
        df['Temp_Col'] = 'All'
        
    # Create a table of unique dimensions - this is what we loop across for each individual forecast
    forecast_dimensions = df.drop(columns = ['ds', 'y']).drop_duplicates()
    
    # Create holiday dataset to pass along into Prophet. This calls the earlier function
    holiday_list = generate_holidays(df = df, fcst_length_year = math.ceil(fcst_length / 365), countries = country_list)
  
    # Create a data frame to store all the forecasts. After each loop, each is placed into this df and each
    # subsequent forecast is appended to the data frame
    output = pd.DataFrame()

    # Print status update
    print('Holidays generated - moving to model fitting')

    #***************************************************************************************** 
    # Loop through each dimension and forecast, placing into the "output" data frame
    #*****************************************************************************************
  
    for i in range(0, forecast_dimensions.shape[0]):

        # Grab the table of all unique dimensions and loop across each row, joining all values from
        # that row against the main dataset. This only returns one unique time series to forecast
        join_dims = forecast_dimensions.iloc[i:i+1,:]
        join_keys = list(join_dims.columns)
        historicals = pd.merge(df, join_dims, on = join_keys)

        #***************************************************************************************** 
        # If enabled, run cross-validation on historicals to pick the best model
        #*****************************************************************************************
        if perform_cross_validation==True:
        
            # Print status update
            print('Running cross-validation')    
        
            # Create parameter grid
            if cross_validation_param_grid==False:
                param_grid = {
                    "changepoint_prior_scale": [0.01, 0.1, 1, 10],
                    "seasonality_prior_scale": [0.01, 0.1, 1, 10],
                    "holidays_prior_scale": [0.01, 0.1, 1, 10],
                    "n_changepoints": [20, 50, 100],
                    "growth": ["linear", "logistic"]}
            else:
                param_grid = cross_validation_param_grid
        
            # Generate all combinations of parameters
            all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
            rmses = []  # Store the RMSEs for each params here

            # Use cross validation to evaluate all parameters
            for params in all_params:
            
                # Create prophet object
                m_cv = Prophet(yearly_seasonality=True, 
                               weekly_seasonality=True, 
                               daily_seasonality=False, 
                               holidays=holiday_list, 
                               uncertainty_samples=False,
                               **params).fit(df)
            
                # Runs cross-validation. It starts with the first 730 days and runs a 365 day forecast
                # and assess the forecast vs actuals. It then moves forward by 182 days and does the same
                df_cv = cross_validation(m_cv, 
                                         initial='730 days', 
                                         period='91 days', 
                                         horizon='365 days',
                                         parallel='processes')
            
                # Converts the cross-validation results into performance metrics
                df_p = performance_metrics(df_cv, rolling_window=1)
                rmses.append(df_p['rmse'].values[0])

            # Find the best parameters
            tuning_results = pd.DataFrame(all_params)
            tuning_results['rmse'] = rmses
            best_params = all_params[np.argmin(rmses)]
        
        #***************************************************************************************** 
        # Build and fit model
        #*****************************************************************************************   
    
        # Print status update
        print('Fitting model')
    
        # Create the prophet model 
        m = Prophet(yearly_seasonality=True, 
                    weekly_seasonality=True, 
                    daily_seasonality=False, 
                    holidays=holiday_list,
                    uncertainty_samples=uncertainty_samples_input)
    
        # Fit the prophet model with the historicals for this particular loop and tuned hyperparameters (if
        # we ran cross-validation)
        if perform_cross_validation==False:
            m.fit(historicals[['ds', 'y']])
        elif perform_cross_validation==True:
            m.fit(historicals[['ds', 'y']], **best_params)

        # Create data frame for forecasts based on the forecast length
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

        # Print status update
        print('Finished loop '+str(i)+' of '+str(forecast_dimensions.shape[0]))

    # change the date format to a readable string
    output.loc[:,'ds'] = output['ds'].astype(str)
  
    # drop the temp col if we made one
    if 'Temp_Col' in output.columns:
        output = output.drop(columns = ['Temp_Col'])

    return output
  

# Forecast 
output = run_forecast(df=input_data, 
                      fcst_length=365, 
                      country_list = ['US', 'CN'],
                      uncertainty_samples_input=0, 
                      perform_cross_validation=True, 
                      cross_validation_param_grid=False)
