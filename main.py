import pandas as pd
from sklearn.tree import DecisionTreeRegressor

melbourn_file_path = '/home/abodo/Documents/ml_data_set/melb_data.csv'

melbourn_data = pd.read_csv(melbourn_file_path)

melbourn_data.columns

# Dote-notation to select a target
y = melbourn_data.Price

# features choosing
melbourn_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

X = melbourn_data[melbourn_features]

# print(X.describe())
print(X.head())

melbourne_model = DecisionTreeRegressor(random_state=1)
model = melbourne_model.fit(X, y)
prediction = melbourne_model.predict(X)
print(model)
print(prediction)
