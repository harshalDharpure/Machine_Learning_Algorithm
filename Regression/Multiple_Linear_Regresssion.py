# Importing the libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder()
X_encoded = onehotencoder.fit_transform(X[:, [3]]).toarray()

# Avoiding the dummy variable Trap
X_encoded = X_encoded[:, 1:]

# Adding encoded data into X
X = np.concatenate((X[:, :3], X_encoded), axis=1)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Making Linear Model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Backward Elimination
# Add a column of 1s for intercept term
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)

# Start backward elimination process
X_opt = X[:, [0, 1, 2, 3, 4]]  # Initial X_opt includes all features
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

# Iterate through features and remove if p-value > 0.05
while np.max(regressor_OLS.pvalues) > 0.05:
    max_p_value_index = np.argmax(regressor_OLS.pvalues)
    X_opt = np.delete(X_opt, max_p_value_index, axis=1)
    regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

print("Backward Elimination:")
print(regressor_OLS.summary())

# Forward Selection
# Start with a single feature and keep adding significant ones
num_features = X.shape[1] - 1  # Exclude the intercept term

X_opt2 = X[:, [0]]  # Start with an empty feature set

for _ in range(num_features):
    p_values = []
    for feature_index in range(1, X.shape[1]):
        if feature_index not in X_opt2:
            current_features = np.append(X_opt2, feature_index)
            regressor_OLS2 = sm.OLS(endog=y, exog=X[:, current_features]).fit()
            p_values.append(regressor_OLS2.pvalues[-1])

    if len(p_values) > 0:
        min_p_value_index = np.argmin(p_values)
        X_opt2 = np.append(X_opt2, min_p_value_index + 1)

    else:
        break

print("Forward Selection:")
print(regressor_OLS2.summary())

# Accuracy on Test Set
y_pred = regressor.predict(X_test)
accuracy = regressor.score(X_test, y_test)
print("Accuracy on Test Set:", accuracy)

# Saving the model as a pickle file
with open('regressor_model.pkl', 'wb') as model_file:
    pickle.dump(regressor, model_file)

# Loading the model and making predictions
with open('regressor_model.pkl', 'rb') as model_file:
    loaded_regressor = pickle.load(model_file)

new_data = np.array([[1, 123456, 78901, 34567, 1, 0]])  # Example new data
new_prediction = loaded_regressor.predict(new_data)
print("Predicted Profit for New Data:", new_prediction)

