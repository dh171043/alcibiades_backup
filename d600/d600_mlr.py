import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.set_option('display.max_columns', None) # view all columns when printing

print("*" * 40)

# houses = pd.read_csv("C:/Users/donovan.a.hall.NREN/PycharmProjects/WGU/d600/houses.csv")
houses = pd.read_csv("D:/programming_workspace/WGU/d600/d600-statistical-data-mining/d600_1/houses.csv")

houses = houses.drop(columns = ['ID'])

dependent_variable = 'Price'
X = houses[['SquareFootage', 'NumBathrooms', 'NumBedrooms',
       'DistanceToCityCenter','RenovationQualityRate','BackyardSpace',
        'PreviousSalePrice','IsLuxury',
        'EmploymentRate', 'CrimeRate', 'Fireplace','HouseColor_Yellow', 'Garage', 'HouseColor_Green', 'Windows', 'Floors',
        'TransportAccess', 'AgeOfHome', 'SchoolRating', 'LocalAmenities', 'HouseColor_Red', 'HouseColor_White', 'PropertyTaxRate']]
y = houses[dependent_variable]
X = sm.add_constant(X)

print(X.columns)

print(X.head())
print(y.head())

# Create a 70/30 split for training and testing, random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training set has {len(X_train)} records.")
print(f"Testing set has {len(X_test)} records.")

X_train.to_csv("D:/programming_workspace/WGU/d600/d600-statistical-data-mining/d600_1/X_train.csv", index=False)
X_test.to_csv("D:/programming_workspace/WGU/d600/d600-statistical-data-mining/d600_1/X_test.csv", index=False)


# Create the OLS model
model = sm.OLS(y_train, X_train)

# Fit the model to the training data
fitted_model = model.fit()

print(fitted_model.summary())




def backward_elimination(X, y, pvalue = 0.05):
    X = X.copy()
    while True:
        model = sm.OLS(y, X).fit()
        p_values = model.pvalues.drop('const')
        max_p = p_values.max()

        if max_p > pvalue:
            highest_value = p_values.idxmax()
            print(f"Highest p value is {highest_value} with a p-value of {max_p: .4f}")
            X = X.drop(columns = [highest_value])
        else:
            print(f"There are no p-values higher than {pvalue}")
            break

    print("\nFinal Model Variables:")
    print(X.columns)
    return X, model
    
X_selected, fitted_model = backward_elimination(X_train, y_train)
print(fitted_model.summary())

X_test_selected = X_test[X_selected.columns]

# Evaluate test data
print('*' *30)
y_pred = fitted_model.predict(X_test_selected)
# Calculate RMSE
mean_price = np.mean(y)
mse_test = mean_squared_error(y_test, y_pred)
rmse_test = np.sqrt(mse_test)
print(f"\nRMSE of the model: {rmse_test: .2f}\nMean of dependent variable: {mean_price: .2f}\nPercentage of RMSE to Mean: {rmse_test/mean_price*100: .2f}%")
