# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
from datetime import datetime, timedelta

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

from greykite.framework.templates.autogen.forecast_config import ForecastConfig, MetadataParam, ModelComponentsParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.utils.result_summary import summarize_grid_search_results
from greykite.framework.constants import MEAN_COL_GROUP, OVERLAY_COL_GROUP
from greykite.framework.input.univariate_time_series import UnivariateTimeSeries

from greykite.algo.changepoint.adalasso.changepoint_detector import ChangepointDetector
from greykite.algo.forecast.silverkite.constants.silverkite_holiday import SilverkiteHoliday
from greykite.algo.forecast.silverkite.constants.silverkite_seasonality import SilverkiteSeasonalityEnum

from greykite.common.constants import TIME_COL
from greykite.common.data_loader import DataLoader
from greykite.common.viz.timeseries_plotting import plot_multivariate, add_groupby_column

from google.colab import auth
import gspread
from google.auth import default

# Build forecasting object
# Comes from here: https://linkedin.github.io/greykite/docs/0.1.0/html/pages/stepbystep/0400_configuration.html
metadata = MetadataParam(
    time_col="ds",     # time column in `df`
    value_col="y",     # value in `df`
    freq='M',
    date_format="%Y-%m-%d",
    anomaly_info=None
)

# Configure seasonality parameters
seasonality_params = dict(
    yearly_seasonality=True,
    quarterly_seasonality=False,
    monthly_seasonality=False,
    weekly_seasonality=True,
    daily_seasonality=False,
)

# Set model parameters: form here (https://github.com/linkedin/greykite/blob/master/docs/nbpages/tutorials/0100_forecast_tutorial.py)
model_components = ModelComponentsParam(
    seasonality=seasonality_params
)

# creates forecasts and stores the result
forecaster = Forecaster()  

# Fit the model and run forecast with input data
result = forecaster.run_forecast_config(
     df=input_data,
     config=ForecastConfig(
         # uses the SILVERKITE model template parameters
         model_template=ModelTemplateEnum.SILVERKITE.name,
         model_components_param=model_components, # Set model components
         forecast_horizon=120,  # forecasts X steps ahead
         coverage=0.95,         # 95% prediction intervals
         metadata_param=metadata
     )
 )

# Plot output
forecaster.forecast_result.forecast.plot()
forecaster.forecast_result.forecast.plot_components()

# Access the result
result.forecast.df['forecast']

# Model summary
result.model[-1].summary()
