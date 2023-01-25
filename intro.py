import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

melbourne_data = pd.read_csv('basic-data-exploration/melb_data.csv')
# melbourne_data.describe()
melbourne_data.columns

# dropna drops missing values
melbourne_data.dropna(axis=0)
y = melbourne_data.Price
X = melbourne_data[['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longitude']]
X.describe()

# Define model, specifying a constant random_state to ensure same results on each run
# melbourne_model = DecisionTreeRegressor(random_state=7)
# Fit model
# melbourne_model.fit(X, y)

# print('Making predictions for the following 5 houses:')
# print(X.head())
# print('The predictions are:')
# print(melbourne_model.predict(X.head()))

# Bad practice: Calculate MAE with same dataset we used for training
# predicted_melbourne_prices = melbourne_model.predict(X)
# print(mean_absolute_error(y, predicted_melbourne_prices))

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(train_X, train_y)
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))