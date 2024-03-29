#************************************************************************************************************
# Purpose:  
#     Run a causal inference on observational data. The purpose is to assess the impact of
#     a particular initiative over a large sample. The script does the following:
#         1. Inputs data: each observation is an "entry", and whether it is subjected to a 
#            particular treatment is shown as a 1 or 0 in the "treatment" column
#         2. Finds controls: For each value that is being treated (treatment == 1), the model
#            uses Nearest Neighbor matching to find control observations that behave similarly.
#            This generates the control group for each treatment observation for causal inference
#         3. Impact Analysis: For each treatment observation, the model runs Google's CausalImpact model, 
#            using that treatment observation as the treatment, and the control observations 
#            from (3) as the control group. The predicted and actual post-treatment uplift is returned
#         4. Summary: Once all the actual and predicted values are returned, the model then calculates
#            a percent uplift for each observation. This is measured as actual value / predicted value.
#            Each date period is then indexed to the treatment period (eg. Treatment period + 1, + 2, etc)
#            and the mean & stdev of impact is measured at each index
#         5. Output: A chart is built looking at the mean uplift and volatility of +/- N periods around
#            the treatment date. This is measured as the uplift of a particular treatment for this metric
#
#     The input data below is generated in the first part, but the actual table should have the following schema:
#         entry: Represents the observation that we are assessing impact of. Can be customer, user, city, etc
#         metric: The metric we are assessing. Typically revenue, page views, margin, cost, or any quantitative metric
#         treatment: A value of 1 or 0 if the particular treatment is present for that time period
#         date: a date value representing the time period at which the above values are tied to
# 
# Source:   
#     Original R package: https://google.github.io/CausalImpact/CausalImpact.html
#     Python Library used here: https://pypi.org/project/pycausalimpact/
# 
#************************************************************************************************************

# Install library
!pip install pycausalimpact

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import datetime
import random
import math
from statsmodels.tsa.arima_process import ArmaProcess
from causalimpact import CausalImpact

################################################################################################
##################################### Generate sample data #####################################
################################################################################################
np.random.seed(12345)
ar = np.r_[1, 0.9]
ma = np.array([1])
arma_process = ArmaProcess(ar, ma)

X1 = random.randint(90,120) + (0.75 + random.random()/2) * arma_process.generate_sample(nsample=100)
X2 = random.randint(90,120) + (0.75 + random.random()/2) * arma_process.generate_sample(nsample=100)
X3 = random.randint(90,120) + (0.75 + random.random()/2) * arma_process.generate_sample(nsample=100)
X4 = random.randint(90,120) + (0.75 + random.random()/2) * arma_process.generate_sample(nsample=100)
X5 = random.randint(90,120) + (0.75 + random.random()/2) * arma_process.generate_sample(nsample=100)
X6 = random.randint(90,120) + (0.75 + random.random()/2) * arma_process.generate_sample(nsample=100)
X7 = random.randint(90,120) + (0.75 + random.random()/2) * arma_process.generate_sample(nsample=100)
X8 = random.randint(90,120) + (0.75 + random.random()/2) * arma_process.generate_sample(nsample=100)
X9 = random.randint(90,120) + (0.75 + random.random()/2) * arma_process.generate_sample(nsample=100)
X10 = random.randint(90,120) + (0.75 + random.random()/2) * arma_process.generate_sample(nsample=100)
Y1 = random.randint(90,120) + (0.75 + random.random()/2) * arma_process.generate_sample(nsample=100)
Y2 = random.randint(90,120) + (0.75 + random.random()/2) * arma_process.generate_sample(nsample=100)
Y3 = random.randint(90,120) + (0.75 + random.random()/2) * arma_process.generate_sample(nsample=100)
Y4 = random.randint(90,120) + (0.75 + random.random()/2) * arma_process.generate_sample(nsample=100)
Y5 = random.randint(90,120) + (0.75 + random.random()/2) * arma_process.generate_sample(nsample=100)

# Uplevel it
Y1[70:] = Y1[70:] + random.randint(1,15)
Y2[60:] = Y2[60:] + random.randint(1,15)
Y3[85:] = Y3[85:] + random.randint(1,15)
Y4[40:] = Y4[40:] + random.randint(1,15)
Y5[55:] = Y5[55:] + random.randint(1,15)

# Create range of dates
datelist = pd.date_range('2021-01-01', periods=100).tolist()

# Create an empty list to store the dataframe rows
rows = []

# Iterate over each array and its corresponding name
for name, array in [("X1", X1),("X2", X2),("X3", X3),("X4", X4),("X5", X5),("X6", X6),("X7", X7),("X8", X8),("X9", X9),("X10", X10),("Y1", Y1),("Y2", Y2),("Y3", Y3),("Y4", Y4),("Y5", Y5)]:
    # Append rows for each value in the array
    for value in array:
        rows.append((name, value))

# Create a dataframe from the rows
df = pd.DataFrame(rows, columns=["entry", "metric"])
df.loc[:,['treatment']] = 0
df.iloc[1070:1099,2] = 1
df.iloc[1160:1199,2] = 1
df.iloc[1285:1299,2] = 1
df.iloc[1340:1399,2] = 1
df.iloc[1455:1499,2] = 1

# Add back dates
df['date'] = np.tile(datelist, df.shape[0] // len(datelist))

###########################################################################################
##################################### Build the model #####################################
###########################################################################################

# Step 1: Parse out observations into treatment and control tables
def parse_observations(data):
    """
    Parses out the treatment and control observations from the input data. It returns two data 
    frames - one with the treatment observations, and one with control observations.
    
    Args:
        data: The data frame used for the causal impact modeling
        
    Returns:
        Two data frames, one of treatment and one of control entries
    
    """
    treatment_table = data.loc[data['treatment'] == 1, ['entry']].drop_duplicates()
    control_table = data.loc[~data['entry'].isin(treatment_table['entry']), ['entry']].drop_duplicates()
    return treatment_table, control_table

# Step 2: Return control entries
def find_control_observations(data, n_controls=5):
    """
    Generates control observations for each treatment observation using Nearest Neighbor
      matching on size, growth, and volatility
    
    Args:
        data: The data frame used for the causal impact modeling
        n_controls: the number of control entries to be selected per treatment entry
        
    Returns:
        A data frame of control observations to be used for calculating the counterfactual
    
    """

    # Parse out treatment and control obs
    treatment_table, control_table = parse_observations(data)
    
    # Build empty df for storing results
    output = pd.DataFrame(columns=['treatment_entry', 'control_entry'])

    # Select control obs for each treatment observation
    for input_entry in enumerate(treatment_table['entry']):
      
      # get treatment stats
      tmp = data.loc[(data['entry'] == input_entry[1]) & (data['treatment'] == 0)].reset_index()
      tmp_slope = tmp.reset_index().groupby(['entry']).apply(lambda tbl: np.polyfit(tbl['index'], tbl['metric'], 1)[0])
      tmp_mean = tmp.groupby(['entry'])['metric'].mean()
      tmp_vol = tmp.groupby(['entry'])['metric'].std()

      # get control stats
      tmp_control = data.loc[(data['entry'].isin(control_table['entry'])) & (data['date'].isin(tmp['date']))]
      tmp_control_slopes = tmp_control.reset_index().groupby(['entry']).apply(lambda tbl_c: np.polyfit(tbl_c['index'], tbl_c['metric'], 1)[0])
      tmp_control_mean = tmp_control.groupby(['entry'])['metric'].mean()
      tmp_control_vol = tmp_control.groupby(['entry'])['metric'].std()

      # Consolidate
      treat = pd.concat([tmp_slope, tmp_mean, tmp_vol], axis=1)
      control = pd.concat([tmp_control_slopes, tmp_control_mean, tmp_control_vol], axis=1)
      combine = pd.concat([treat, control])
      combine.columns = ['slope', 'mean', 'volatility']

      # Fit and run knn algo
      nbrs = NearestNeighbors(n_neighbors=n_controls+1, algorithm='ball_tree').fit(combine.values)
      distances, indices = nbrs.kneighbors(combine.values)

      # Create a DataFrame to store the results
      output_tmp = pd.DataFrame({"treatment_entry": input_entry[1], 'control_entry': combine.index[indices[0]][1:]})
      output = pd.concat([output, output_tmp])
    
    # Return output
    return output

# Step 3: Run causal inference model
def run_causal_impact(data, n_controls = 5, window = 10):
    """
    Runs the causal impact model. It first brings in the treatment and control tables from
    an earlier function. It then creates an empty data frame of values to export later. The
    function then loops through each treatment entry, doing the following:
        1. Bring in a set of control entries to match against
        2. Pull all records for the treatment entry and the control entries
        3. Create columns of dates, whether the treatment is there or not, and index of 
            which dates are treated
        4. Pivot the treatment and control datasets and merge them into one
        5. Run the merged dataset through CausalImpact, using the index above to guide when
            the treatment vs control dates are
        6. Create a table of date, entry, actual_value, counterfactual_value, and treatment 
            columns, and add in an indexed date column
        7. Stack the table above with the empty data frame created at the start and continue
            on to the next entry and do the same thing
    
    Args:
        data: The data frame used for the causal impact modeling
        n_controls: the number of control entries to be selected per treatment entry
        window: the number of periods before / after to plot the graph
        
    Returns:
        A data frame of actual and counterfactual values for each entry by date
        A plot to show the impact values
    
    """
    # Pull treatment and control tables
    treatment_table, control_table = parse_observations(data)
    
    # Create temporary df to hold output
    output = pd.DataFrame(columns=['date', 'treatment', 'entry', 'y', 'y_pred', 'indexed_date'])

    # Loop through each treatment entry
    for entry in treatment_table['entry']:

        # Pull control observations and intial treatment data
        treat_control_nn = find_control_observations(df, n_controls)
        control_obs = treat_control_nn.loc[treat_control_nn['treatment_entry'] == entry, ['control_entry']]
        treatment_data = data.loc[data['entry'] == entry].reset_index(drop=True)

        # Get treatment dates and the treatment column
        trt_col = treatment_data['treatment']
        date_col = treatment_data['date']

        # Get treatment index
        idx = treatment_data.loc[treatment_data['treatment'] == 1].index

        # Pivot treatment and control data
        treatment_data = treatment_data.pivot(index='date', columns='entry', values='metric').reset_index(drop=True)
        control_data = data.loc[data['entry'].isin(control_obs['control_entry'])]
        control_data = control_data.pivot(index='date', columns='entry', values='metric').reset_index(drop=True)

        # get pre and post periods
        pre_period = [int(0), int(idx.min()-1)]
        post_period = [int(idx.min()), int(treatment_data.shape[0]-1)]

        # Create CausalImpact input data
        input_df = pd.concat([treatment_data, control_data], axis = 1)
        ci = CausalImpact(input_df, pre_period, post_period, prior_level_sd=None)

        # Compile summary metrics
        combine = pd.DataFrame({'date': date_col, 'treatment':trt_col, 'entry': entry, 'y': ci.data.iloc[:,0], 'y_pred': ci.inferences['preds']}, 
                               columns=['date', 'treatment', 'entry', 'y', 'y_pred']).reset_index(drop=True)
        
        # Add a column for indexed date so we know the impact X days before / after adoption
        combine['indexed_date'] = combine['date'] - combine.loc[combine['treatment'] == 1, ['date']].min()[0]
        combine['indexed_date'] = [i.days for i in combine['indexed_date']]

        # Union with main df
        output = pd.concat([output, combine])

    # Create columns for delta and percentage delta
    output['delta'] = output['y'] - output['y_pred']
    output['delta_pct'] = output['y'] / output['y_pred'] - 1

    # Subset between the upper and lower range and plot it
    output_subset = output[output['indexed_date'].between(-window, window)]
    subset_df = pd.DataFrame({
        'index_date': output_subset.groupby('indexed_date')['indexed_date'].max(),
        'delta': output_subset.groupby('indexed_date')['delta_pct'].mean(),
        'stdev': output_subset.groupby('indexed_date')['delta_pct'].std()
    }, columns = ['index_date', 'delta', 'stdev']).reset_index(drop=True)
    
    # Calc lower and upper CI
    subset_df['lower'] = subset_df['delta'] - 1.96 * subset_df['stdev'] / math.sqrt(len(output_subset['entry'].unique()))
    subset_df['upper'] = subset_df['delta'] + 1.96 * subset_df['stdev'] / math.sqrt(len(output_subset['entry'].unique()))
        
    # Create plot
    fig, ax = plt.subplots()
    
    # Plot the scatter plot with different colors based on a column
    ax.plot(subset_df['index_date'], subset_df['delta'], color = 'cornflowerblue')
    ax.plot(subset_df['index_date'], subset_df['lower'], color = 'lightsteelblue')
    ax.plot(subset_df['index_date'], subset_df['upper'], color = 'lightsteelblue')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.fill_between(subset_df['index_date'], subset_df['lower'], subset_df['upper'], color='lightblue', alpha=0.5)
    ax.axhline(y=0, color='red', linestyle='--', lw = 0.5)
    ax.axvline(x=0, color='grey', linestyle='--', lw = 0.5)
        
    # Show the plot
    plot = plt.show()

    # Return values
    return output, subset_df, plot

#################################################################################
########################### Run the model #######################################
#################################################################################

df = pd.read_csv('input_df.csv')
output_df, subset_df, plot = run_causal_impact(df)
