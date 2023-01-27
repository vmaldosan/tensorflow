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

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    return mean_absolute_error(val_y, preds_val)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print('Max leaf nodes: %d \t\t MAE: %d' %(max_leaf_nodes, my_mae))

# Shorter version
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in [5, 50, 500, 5000]}
best_tree_size = min(scores, key=scores.get)
print(best_tree_size)
