import pandas as pd
from sklearn.tree import DecisionTreeRegressor

melbourne_data = pd.read_csv('basic-data-exploration/melb_data.csv')
# melbourne_data.describe()
melbourne_data.columns

# dropna drops missing values
melbourne_data.dropna(axis=0)
y = melbourne_data.Price
X = melbourne_data[['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longitude']]
X.describe()

# Define model, specifying a constant random_state to ensure same results on each run
melbourne_model = DecisionTreeRegressor(random_state=7)
# Fit model
melbourne_model.fit(X, y)

print('Making predictions for the following 5 houses:')
print(X.head())
print('The predictions are:')
print(melbourne_model.predict(X.head()))