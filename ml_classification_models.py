##############################################################################################
# Call python libraries
##############################################################################################
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, cross_validate
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB
from sklearn import preprocessing, tree, datasets, svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score, plot_confusion_matrix, r2_score

##############################################################################################
# Read in data
##############################################################################################

# read in white and red wine data
header_names_list = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol", "quality"]
white_wine_data = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/winequality-white.csv", header = None, names = header_names_list)
red_wine_data = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/winequality-red.csv", header = None, names = header_names_list)

# Add columns and combine
white_wine_data["type"] = 'white'
red_wine_data["type"] = 'red'
wine_data = pd.concat([white_wine_data, red_wine_data], axis = 0)

# Encode vertical so it works with models
features_to_encode = [i for i in wine_data.dtypes[wine_data.dtypes == 'object'].index]

# Create constructor to handle categorical variables
col_trans = make_column_transformer((OneHotEncoder(),features_to_encode), remainder = "passthrough")

# Load predictor and target variables into x and y
x_df = wine_data.drop(columns = output)
y_df = wine_data[output]

# Split training and test for x and y
x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size = 0.2, stratify = y_df) # stratify ensures balance in response variables in train & test

# Get output variable
output = 'quality'

##############################################################################################
# CART Trees
##############################################################################################

clf_cart = DecisionTreeClassifier()

# Buld parameter ranges
param_grid_cart = {
    'max_depth': range(3, 8),
    'ccp_alpha': np.arange(0.01, 0.3, 0.01)
    }

# build grid of inputs and run through the possible ranges
grid_clf_cart = GridSearchCV(clf_cart, param_grid_cart, cv=10, n_jobs = -1)
grid_clf_cart.fit(x_test, y_train)

# Build CART tree classifier
cart = DecisionTreeClassifier(
    max_depth = grid_clf_cart.best_params_['max_depth'],
    ccp_alpha = grid_clf_cart.best_params_['ccp_alpha']
    )

# Fit the RF model (using the input variables) and predict
cart.fit(x_test, y_train)
y_pred_cart = cart.predict(x_test)

# Plot the output
fig = plt.figure(figsize=(20, 8))
cols = pd.get_dummies(x_train).columns
plot_tree(cart, filled=True, rounded=True, class_names = cols, fontsize=10)
plt.show()

##############################################################################################
# Random Forest Model
##############################################################################################

# Setup cross validation
clf_rf = RandomForestClassifier()

# Buld parameter ranges
param_grid_rf = {
    'max_features': range(4, 10),
    'n_estimators': range(300, 800, 100)
    }

# build grid of inputs and run through the possible ranges
grid_clf_rf = GridSearchCV(clf_rf, param_grid_rf, cv=10, n_jobs = -1)
grid_clf_rf.fit(x_test, y_train)

# Build random forest classifier
rf_model = RandomForestClassifier(
    n_estimators = grid_clf_rf.best_params_['n_estimators'],
    max_features = grid_clf_rf.best_params_['max_features'],
    n_jobs = -1)

# Fit the RF model (using the input variables) and predict
rf = make_pipeline(col_trans, rf_model)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
accuracy_score(y_test, y_pred_rf)

##############################################################################################
# Gradient Boosting Trees
##############################################################################################

# Setup cross validation
clf_gb = GradientBoostingClassifier()

# Buld parameter ranges
param_grid_gb = {
    'learning_rate': np.arange(0.05, 1.0, 0.05)
    }

# build grid of inputs and run through the possible ranges
grid_clf_gb = GridSearchCV(clf_gb, param_grid_gb, cv=10, n_jobs = -1)
grid_clf_gb.fit(x_test, y_train)

# Build gradient boosted trees
gb_model = GradientBoostingClassifier(
    n_estimators = 100,
    learning_rate = grid_clf_gb.best_params_['learning_rate'])

# Fit the GB model (using the input variables) and predict
gb = make_pipeline(col_trans, gb_model)
gb.fit(x_train, y_train)
y_pred_gb = gb.predict(x_test)

##############################################################################################
# XGBoost Model
##############################################################################################

# Fit model
cv_xbg = XGBClassifier(tree_method="gpu_hist")

# Fit cross-validation
param_grid_xgb = {
    'learning_rate':np.arange(0.1, 0.6, 0.1), # eta
    'max_depth': range(1, 6, 1), # max depth of a tree
    'colsample_bytree': np.arange(0.6, 1, 0.2),
    'subsample': np.arange(0.25, 1.25, 0.25)
    }

# Run cross validation
grid_clf_xbg = GridSearchCV(cv_xbg, param_grid_xgb, cv=3, n_jobs=-1)
grid_clf_xbg.fit(x_test, y_train)

# Fit final model
xbg_model = XGBClassifier(
    tree_method="gpu_hist",
    learning_rate = grid_clf_xbg.best_params_['learning_rate'],
    max_depth = grid_clf_xbg.best_params_['max_depth'],
    colsample_bytree = grid_clf_xbg.best_params_['colsample_bytree'],
    subsample = grid_clf_xbg.best_params_['subsample'])

xbg_model.fit(x_test, y_train)
y_pred_xgb = xbg_model.predict(x_test)
accuracy_score(y_test, y_pred_xgb)
