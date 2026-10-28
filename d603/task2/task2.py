import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_columns", None)  # view all columns when printing
spacer = "*" * 20
# churn = pd.read_csv("C:/Users/donovan.a.hall.NREN/PycharmProjects/WGU/d603/task1/churn_clean.csv")
# columns_and_dtypes = pd.read_excel("C:/Users/donovan.a.hall.NREN/PycharmProjects/WGU/d603/task1/task1_dtypes.xlsx")
churn = pd.read_csv("C:/alcibiades_vscode/WGU/d603/task1/churn_clean.csv")
columns_and_dtypes = pd.read_excel(
    "C:/alcibiades_vscode/WGU/d603/task1/task1_dtypes.xlsx"
)


def header_txt(message):
    output = f"\n{spacer} {message} {spacer}\n"
    print(output)


# EDA steps
def EDA(dataframe):
    """
    collection of views for initial look at data
    """
    print(spacer, "HEAD", spacer)
    print(dataframe.head())  # view the data
    print("*" * 20, "DESCRIBE", "*" * 20)
    print(dataframe.describe())  # look at numerical columns
    print("*" * 20, "SHAPE", "*" * 20)
    print(dataframe.shape)  # see number of rows/columns
    print("*" * 20, "ISNULL", "*" * 20)
    print(
        dataframe.isnull().sum()
    )  # look for blank cells (InternetService has 2129 nulls)
    print("*" * 20, "ISDUP", "*" * 20)
    print(dataframe.duplicated().sum())  # look for duplicated rows (no dups)
    print("*" * 20, "DTYPES", "*" * 20)
    print(
        dataframe.dtypes
    )  # look at data types (fireplace, color, garage are the only non-numeric columns, lets look at those)


# fix nulls in InternetService
churn = churn.fillna("No Service")
EDA(churn)

# keeping the variables listed in D2
keep = columns_and_dtypes["Variable"]
churn = churn[keep]
print("*" * 20, "kept variables", "*" * 20)
print(churn)


# encoding churn dataframe
def boolean_encode(dataframe, columns):
    for i in columns:
        print(dataframe[i].head())
        print(f"The data type is {dataframe[i].dtype}")
        dataframe[i] = dataframe[i].replace({"Yes": 1, "No": 0})
    return dataframe


def encoding(dataframe, list_of_columns):
    churn_encoded = pd.get_dummies(dataframe, columns=list_of_columns, dtype=int)
    return churn_encoded


numeric_vars = columns_and_dtypes[columns_and_dtypes["DTYPE"] == "numerical"]
numeric_vars = numeric_vars["Variable"]
categorical_vars = columns_and_dtypes[columns_and_dtypes["DTYPE"] == "categorical"]
categorical_vars = categorical_vars["Variable"]
print(categorical_vars)

var_bool = []
var_cat = []
for i in categorical_vars:
    unique = set(churn[i])
    if "Yes" in unique:
        var_bool.append(i)
    else:
        var_cat.append(i)
    print(f"The unique values in {i} are {unique}")
# item1-8 do need encoded
list_items = [""]
list_toencode = []
for i in var_cat:
    if i not in list_items:
        list_toencode.append(i)
header_txt("")
header_txt("")
print(f"Boolean variables: {var_bool}")
print(f"Categorical variables: {var_cat}")
print(f"Categorical variables to encode: {list_toencode}")

print("*" * 20, "encoded variables", "*" * 20)
churn_encoded = encoding(churn, list_toencode)
churn_encoded_columns = churn_encoded.columns
print(churn_encoded_columns)  # check for proper one-hot encoding
print(churn_encoded.head())
print("*" * 20, "new code", "*" * 20)
print(churn_encoded_columns)
for i in churn_encoded_columns:
    print(i)
churn_encoded = boolean_encode(churn_encoded, churn_encoded_columns)
print(churn_encoded.head())
print(churn_encoded.dtypes)

# churn_encoded.to_csv("C:/Users/donovan.a.hall.NREN/PycharmProjects/WGU/d603/task2/churn_encoded.csv", index=False)
churn_encoded.to_csv(
    "C:/alcibiades_vscode/WGU/d603/task2/churn_encoded.csv", index=False
)

# scaling the data
scaler = StandardScaler()
processed = scaler.fit_transform(churn_encoded)
processed_df = pd.DataFrame(processed, columns=churn_encoded.columns)

print(header_txt("scaled data"))
print(processed_df.head())

# dendrogram
header_txt("Constructing Dendrogram")
linked = linkage(processed_df, method="ward")
plt.figure(figsize=(12, 8))
dendrogram(
    linked,
    orientation="top",
    labels=None,
    distance_sort="descending",
    show_leaf_counts=True,
)
title = "Hierarchical Clustering Dendrogram using Ward Linkage"
plt.title(title)
plt.xlabel("Data Points")
plt.ylabel("Euclidian Distance")
# plt.savefig(f"C:/Users/donovan.a.hall.NREN/PycharmProjects/WGU/d603/task2/{title}")
plt.savefig(f"C:/alcibiades_vscode/WGU/d603/task2/{title}")


linkage_df = pd.DataFrame(
    linked,
    columns=["Cluster ID 1", "Cluster ID 2", "Distance", "Num Points"],
    index=[f"Merge {i+1}" for i in range(linked.shape[0])],
)

header_txt("LINKAGE DATAFRAME")
print(linkage_df.tail(20))


# agglomerative clustering
header_txt("Begin Agglomerative Clustering")
n_clusters = 3
model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
labels = model.fit_predict(processed_df)

churn_encoded["Cluster"] = labels
processed_df["Cluster"] = labels

header_txt("Clustering Complete")


# analyze PREP
def num_cat_split(df):
    numeric_cols = []
    categorical_cols = []
    for col in df.columns:
        if col == "Cluster":
            continue
        if set(churn_encoded[col].unique()).issubset({0, 1}):
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)
    return numeric_cols, categorical_cols


header_txt("Profile Analysis")
cluster_profiles = churn_encoded.groupby("Cluster").mean()
scaled_cluster_profiles = processed_df.groupby("Cluster").mean()

header_txt("UNSCALED CLUSTER PROFILES")
print(cluster_profiles)
# cluster_profiles.to_csv("C:/Users/donovan.a.hall.NREN/PycharmProjects/WGU/d603/task2/cluster_profiles.csv")
cluster_profiles.to_csv("C:/alcibiades_vscode/WGU/d603/task2/cluster_profiles.csv")


header_txt("Cluster Visualizations via Heatmap")

unscaled_ranges = cluster_profiles.max() - cluster_profiles.min()
important_unscaled_variables = unscaled_ranges.sort_values(ascending=False)

scaled_ranges = scaled_cluster_profiles.max() - scaled_cluster_profiles.min()
important_scaled_variables = scaled_ranges.sort_values(ascending=False)

# calling function to split numeric from categorical data
unscaled_num, unscaled_cat = num_cat_split(churn_encoded)
numeric_unscaled_importance = important_unscaled_variables.loc[
    unscaled_num
].sort_values(ascending=False)
categorical_unscaled_importance = important_unscaled_variables.loc[
    unscaled_cat
].sort_values(ascending=False)
header_txt("NUMERIC")
print(f"Numeric Columns:\n {numeric_unscaled_importance}")
print(numeric_unscaled_importance.head())
header_txt("CATEGORICAL")
print(f"Categorical Columns:\n {categorical_unscaled_importance}")
print(categorical_unscaled_importance.head())
header_txt("SCALED")
print(important_scaled_variables)
# important_scaled_variables.to_csv("C:/Users/donovan.a.hall.NREN/PycharmProjects/WGU/d603/task2/important_scaled_variables.csv")
important_scaled_variables.to_csv(
    "C:/alcibiades_vscode/WGU/d603/task2/important_scaled_variables.csv"
)


# from my analysis there is a huge dropoff after 0.9
def heatmap_fctn(scaled_df, cutoff, title_text):
    important_scaled_variable_range = scaled_df > float(cutoff)
    scaled_df = scaled_df[important_scaled_variable_range]
    important_variables_for_heatmap = scaled_df.index

    header_txt("VARIABLES BEING USED IN HEATMAP")
    print(important_variables_for_heatmap)

    heatmap_data = scaled_cluster_profiles[important_variables_for_heatmap]
    # heatmap_data.to_csv("C:/Users/donovan.a.hall.NREN/PycharmProjects/WGU/d603/task2/heatmap_data.csv")
    heatmap_data.to_csv("C:/alcibiades_vscode/WGU/d603/task2/heatmap_data.csv")

    plt.figure(figsize=(16, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis")
    title = f"Heatmap of Cluster Profiles from {title_text} and up"
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Cluster")
    # plt.savefig(f"C:/Users/donovan.a.hall.NREN/PycharmProjects/WGU/d603/task2/{title}")
    plt.savefig(f"C:/alcibiades_vscode/WGU/d603/task2/{title}")


for i, j in [(0.65, "all"), (1, "top")]:
    heatmap_fctn(important_scaled_variables, i, j)

