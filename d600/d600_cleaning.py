#!/usr/bin/env python3
 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None) # view all columns when printing

# houses = pd.read_csv("C:/Users/donovan.a.hall.NREN/PycharmProjects/WGU/d600/D600 Task 1, 2 and 3 Dataset Housing Information.csv")
houses = pd.read_csv("D:/programming_workspace/WGU/d600/d600-statistical-data-mining/d600_1/D600 Task 1, 2 and 3 Dataset Housing Information.csv")
# EDA steps
def EDA(dataframe):
    """
    collection of views for initial look at data
    """
    print("*"*20, "EDA", "*"*20)
    print(houses.head()) # view the data
    print(houses.describe()) # look at numerical columns
    print(houses.shape) # see number of rows/columns
    print(houses.isnull().sum()) # look for blank cells (no nulls)
    print(houses.duplicated().sum()) #look for duplicated rows (no dups)
    print(houses.dtypes) # look at data types (fireplace, color, garage are the only non-numeric columns, lets look at those)

    non_num = houses[["Fireplace", "HouseColor", "Garage"]]
    print(non_num.head()) # Fireplace and Garage are both yes/no, so lets change them to 1/0

    # look for unique values
    set_fireplace = set(non_num["Fireplace"])
    set_garage = set(non_num["Garage"])
    print(set_fireplace)
    print(set_garage) # both columns only have "Yes" and "No", so a simple replace should work here.

def boolean_encode(column):
    houses[column] = houses[column].str.lower().replace({'yes':1, 'no':0}) # makes all strs lowercase to improve scalability, then replaces yes:1, no:0

def encoding(dataframe):
    columns_to_convert = ['Fireplace', 'Garage', 'HouseColor']
    for i in columns_to_convert:
        houses = pd.get_dummies(dataframe, dtype = bool, columns = [i], prefix = i, drop_first = True)
        boolean_encode(i)
    dummy_var = ['HouseColor_Green', 'HouseColor_Red', 'HouseColor_White', 'HouseColor_Yellow']
    for column in dummy_var:
        houses[column] = houses[column].astype(bool).astype(int)
    return houses

# remove unwanted columns (must keep at least 10 independant variables)
def drop_unwanted(houses, columns_to_drop):
    houses = houses.drop(columns_to_drop, axis = 1)
    print(houses.columns)

    return houses

def categorical_viz(df):
    non_num = ["Fireplace", "IsLuxury", "Garage", "Floors", "HouseColor"]
    for i in non_num:
        nonnum_counts = pd.Series(df[i].value_counts())
        
        def autopct_format(pct):# receives the percentage for the slice
            total = sum(nonnum_counts) # total number of rows
            count = int(round(pct*total/100))# calutaes absolute count
            return f"{pct:.1f}%\n({count})"# returns formatted label


        plt.pie(
            nonnum_counts, 
            labels = nonnum_counts.index, 
            autopct = autopct_format, 
            startangle = 90
        )
        plt.title(i)
        # plt.show()
        plt.savefig(f"D:/programming_workspace/WGU/d600/d600-statistical-data-mining/d600_1/task1_visuals/Distribution of {i}")
        df.boxplot(column='Price', by=i)
        plt.title(f"Price Distribution by {i}")
        plt.suptitle("")
        plt.xlabel(i)
        plt.ylabel("Price")
        # plt.show()
        plt.savefig(f"D:/programming_workspace/WGU/d600/d600-statistical-data-mining/d600_1/task1_visuals/Price Distribution by {i}")


categorical_viz(houses)
EDA(houses)
houses = encoding(houses)
# houses = drop_unwanted(houses, columns_to_drop=['SchoolRating', 'EmploymentRate', 'RenovationQualityRate', 'TransportAccess', 'Fireplace', 'HouseColor'])

print(houses.columns)

# houses.to_csv("C:/Users/donovan.a.hall.NREN/PycharmProjects/WGU/d600/houses.csv", index=False)
houses.to_csv("D:/programming_workspace/WGU/d600/d600-statistical-data-mining/d600_1/d600_1houses.csv", index=False)

print(houses.head())