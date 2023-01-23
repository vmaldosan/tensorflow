import pandas as pd

melbourne_data = pd.read_csv('basic-data-exploration/melb_data.csv')
melbourne_data.describe()

home_data = pd.read_csv('basic-data-exploration/train.csv')
home_data.describe()

