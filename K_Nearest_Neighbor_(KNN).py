import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Load the Dataset
data = pd.read_csv("country_vaccinations.csv")

# Step 2: Preprocess the Dataset (Drop samples with missing values)
data = data.dropna()

# Step 3: Filter Data for Pakistan
data_pakistan = data[data['country'] == 'Pakistan']

# Step 4: Split the Dataset
X = data_pakistan[['total_vaccinations']]  # Feature
y = data_pakistan['people_vaccinated']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the KNN Regressor
k = 3  # Number of neighbors
knn = KNeighborsRegressor(n_neighbors=k)
knn.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = knn.predict(X_test)

# Step 7: Model Evaluation
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Step 8: Visualization
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Total Vaccinations')
plt.ylabel('People Vaccinated')
plt.title('KNN Regression: Total Vaccinations vs People Vaccinated')
plt.legend()
plt.show()