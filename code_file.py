import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# dataset (15 houses)
data = {
    'Size': [850, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 2000, 2200, 2500, 3000],
    'Bedrooms': [2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6],
    'Age': [20, 18, 15, 10, 8, 7, 5, 5, 4, 3, 3, 2, 1, 1, 1],
    'Price': [150000, 160000, 180000, 200000, 220000, 240000, 260000, 280000, 300000, 
              320000, 340000, 360000, 400000, 450000, 500000]
}

df = pd.DataFrame(data)


X = df[['Size', 'Bedrooms', 'Age']]
y = df['Price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


# Predictions for test set
y_predt = model.predict(X_test)
print("Predicted prices for test set:", y_predt)


# Evaluate the model
mse = mean_squared_error(y_test, y_predt)

print("Intercept (b):", model.intercept_)
print("Slopes (m1, m2, m3):", model.coef_)
print("Mean Squared Error:", mse)

# Predict price for a new house

sqft=int(input("Enter house size (sqft): "))
bedrooms=int(input("Enter number of bedrooms: "))       
age=int(input("Enter house age (years): "))
y_pred = model.predict([[sqft, bedrooms, age]])
print("Predicted Price: $",y_pred[0])
