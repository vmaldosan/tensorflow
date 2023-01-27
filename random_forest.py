import pandas as pd
from sklearn.emsemble import RandomForestRegressor
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

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

melbourne_model = RandomForestRegressor(random_state=1)
melbourne_model.fit(train_X, train_y)
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
