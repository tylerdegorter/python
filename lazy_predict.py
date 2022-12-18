pip install lazypredict

# models
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

# Get output variable
output = 'quality'

# read in white and red wine data
header_names_list = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol", "quality"]
white_wine_data = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/winequality-white.csv", header = None, names = header_names_list)
red_wine_data = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/winequality-red.csv", header = None, names = header_names_list)

# Add columns and combine
white_wine_data["type"] = 'white'
red_wine_data["type"] = 'red'
wine_data = pd.concat([white_wine_data, red_wine_data], axis = 0)
wine_data = pd.get_dummies(wine_data)

# Load predictor and target variables into x and y
x_df = wine_data.drop(columns = output)
y_df = wine_data[output]

# Split training and test for x and y
x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size = 0.2, stratify = y_df) # stratify ensures balance in response variables in train & test

# Build model
lazy_clf = LazyRegressor(verbose = 0, ignore_warnings = True, 
                         predictions = False, custom_metric = None)
models, predictions = lazy_clf.fit(x_train, x_test, y_train, y_test)

# print output
if 'R-Squared' in models.columns:
    print(models[['R-Squared', 'Adjusted R-Squared', 'Time Taken']].sort_values(by = ["R-Squared"], ascending = False))
elif 'Accuracy' in models.columns:
    print(models[['Accuracy', 'ROC AUC', 'Time Taken']].sort_values(by = ["Accuracy"], ascending = False))

# grab best model
