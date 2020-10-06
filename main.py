import pandas as pd
from sklearn.tree import DecisionTreeRegressor

melbourn_file_path = '/home/abodo/Documents/ml_data_set/melb_data.csv'

melbourn_data = pd.read_csv(melbourn_file_path)

melbourn_data.columns

# Dote-notation to select a target# Filter rows with missing price values
filtered_melbourn_data = melbourn_data.dropna(axis=0)
# Choose target and features
y = filtered_melbourn_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourn_data[melbourne_features]
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
print(melbourne_model.fit(X, y))

from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
print(mean_absolute_error(y, predicted_home_prices))

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
melbourn_model = DecisionTreeRegressor()
# Fit model
melbourn_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourn_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))




# features choosing
# melbourn_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
# X = melbourn_data[melbourn_features]

# print(X.describe())
# print(X.head())

# melbourne_model = DecisionTreeRegressor(random_state=1)
#model = melbourne_model.fit(X, y)
# prediction = melbourne_model.predict(X)
# print(model)
# print(prediction)
