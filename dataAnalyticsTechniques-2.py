#!/usr/bin/env python
# coding: utf-8

# # **CREDITCARD FRAUD DATASET**

# **IMPORTING THE DATASET INTO THE WORKSPACE**

# In[ ]:


import pandas as pd #Library to read and process the csv file
creditcard_df = pd.read_csv('creditcard.csv') # Loading the dataset into the dataframe
print(creditcard_df) # Viewing the dataframe


# **EXPLORATORY DATA ANALYSIS**

# In[ ]:


creditcard_df.head(5) # Viewing the first 5 rows


# In[ ]:


creditcard_df.tail(5) # Viewing the last 5 rows


# In[ ]:


creditcard_df.shape # Total number of rows and columns


# In[ ]:


creditcard_df.info() # Summary of the dataframe (data type of each column, memory usage, etc)


# In[ ]:


creditcard_df.describe() # Descriptive statistics (central tendency and dispersion) of the dataframe


# *** Identifying Missing Values And Replacing With Their Median Values***

# In[ ]:


missing=creditcard_df[creditcard_df.isnull().any(axis=1)] # Identify columns with missing values
print (missing.to_string())


# In[ ]:


creditcard_df.isna().sum().sum() # Total number of missing values in the dataframe


# In[ ]:


null = creditcard_df["V14"].median()
creditcard_df["V14"] = creditcard_df["V14"].fillna(null)
null = creditcard_df["V15"].median()
creditcard_df["V15"] = creditcard_df["V15"].fillna(null)
null = creditcard_df["V16"].median()
creditcard_df["V16"] = creditcard_df["V16"].fillna(null)
null = creditcard_df["V17"].median()
creditcard_df["V17"] = creditcard_df["V17"].fillna(null)
null = creditcard_df["V18"].median()
creditcard_df["V18"] = creditcard_df["V18"].fillna(null)
null = creditcard_df["V19"].median()
creditcard_df["V19"] = creditcard_df["V19"].fillna(null)
null = creditcard_df["V20"].median()
creditcard_df["V20"] = creditcard_df["V20"].fillna(null)
null = creditcard_df["V21"].median()
creditcard_df["V21"] = creditcard_df["V21"].fillna(null)
null = creditcard_df["V22"].median()
creditcard_df["V22"] = creditcard_df["V22"].fillna(null)
null = creditcard_df["V23"].median()
creditcard_df["V23"] = creditcard_df["V23"].fillna(null)
null = creditcard_df["V24"].median()
creditcard_df["V24"] = creditcard_df["V24"].fillna(null)
null = creditcard_df["V25"].median()
creditcard_df["V25"] = creditcard_df["V25"].fillna(null)
null = creditcard_df["V26"].median()
creditcard_df["V26"] = creditcard_df["V26"].fillna(null)
null = creditcard_df["V27"].median()
creditcard_df["V27"] = creditcard_df["V27"].fillna(null)
null = creditcard_df["V28"].median()
creditcard_df["V28"] = creditcard_df["V28"].fillna(null)
null = creditcard_df["Amount"].median()
creditcard_df["Amount"] = creditcard_df["Amount"].fillna(null)
null = creditcard_df["Class"].median()
creditcard_df["Class"] = creditcard_df["Class"].fillna(null)


# In[ ]:


missing=creditcard_df.isnull().sum() # Confirming if there are still missing values or not
print(missing)


# *** Handling Redundant Data***

# In[ ]:


redundant = creditcard_df[creditcard_df.duplicated()] # Identifying redundant data
print (redundant.to_string())


# In[ ]:


creditcard_df.duplicated().sum() # Total number of redundant data


# In[ ]:


card=creditcard_df.drop_duplicates(inplace=True)# Dropping redundant data
print(card)


# **INTEGRATING COLUMNS WITH SIMILAR FEATURES**

# In[ ]:


data1 = {
    'Time': creditcard_df['Time'],
    'Amount': creditcard_df['Amount']
}
data2 = {
    'Time': creditcard_df['Time'],
    'Class': creditcard_df['Class']
}

# Creating The Dataframes
df1 = pd.DataFrame(data1)
print(df1.to_string())
df2 = pd.DataFrame(data2)
print(df2.to_string())

# Merging The DataFrames on 'Time' Column
merged_df = pd.merge(df1, df2, on='Time')

print(merged_df) # Print the integrated DataFrame


# **SPLIT INTO TRAIN AND TEST DATASETS**

# In[ ]:


from sklearn.model_selection import train_test_split #Function to split, test and train the dataset
train_df, test_df = train_test_split(creditcard_df, test_size=0.3, random_state=42)


# In[ ]:


train_df.head(5)


# In[ ]:


train_df.shape


# In[ ]:


test_df.head(5)


# In[ ]:


test_df.shape


# # **BREAST CANCER DATASET**

# In[ ]:


import pandas as pd #Library to read and process the csv file
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
cancer_df = pd.read_csv(url, header=None) # Loading the dataset into the dataframe
print(cancer_df) # Viewing the dataframe


# **Explorative Data Analysis**

# In[ ]:


cancer_df.head(5) #First 5 rows


# In[ ]:


cancer_df.tail(5) #Last 5 rows


# In[ ]:


cancer_df.shape #Total number of rows and column


# In[ ]:


cancer_df.info() # Summary of the dataframe (data type of each column, memory usage, etc)


# In[ ]:


cancer_df.describe() # Descriptive statistics (central tendency and dispersion) of the dataframe


# **Data Preprocessing**

# In[ ]:


cancer_df.isna().sum().sum() # Total number of missing values in the dataframe


# In[ ]:


cancer_df.duplicated().sum() # Total number of redundant data


# **Data Standardization**

# In[ ]:


print(cancer_df.columns) #Viewing the names of the columns

cancer_df.iloc[:, 1] = cancer_df.iloc[:, 1].replace({"M": 0, "B": 1}) # Coverting non-numeric values in column1 to numeric
cancer_df.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler #Class from the sklearn.preprocessing module to standardize the dataset

scaler = StandardScaler() #Initializing the standardScaler object

scaled_data = scaler.fit_transform(cancer_df) #Fitting and transforming the data
print (scaled_data)


# **Principal Component Analysis (PCA)**

# In[ ]:


from sklearn.decomposition import PCA #Class from the sklearn.decomposition module for conducting PCA
import matplotlib.pyplot as plt #Library to carryout visualizations(plots)


# In[ ]:


#Replacing the values in column1
cancer_df[1] = cancer_df[1].replace({0: "Malignant", 1: "Benign"})

# Drop the second column of the cancer_df DataFrame
X = cancer_df.drop(columns=[1])  # Features

# Extract the target variable from the second column
y = cancer_df[1]  # Target variable

# Perform PCA on the features
pca = PCA(n_components=2)  # Initialize PCA with 2 components
X_pca = pca.fit_transform(X)  # Fit and transform the data

# Create a new DataFrame with the PCA results and the target variable
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
final_df = pd.concat([pca_df, y], axis=1)

# Print the final DataFrame
print(final_df)


# In[ ]:


# Extract unique values from the target column
target_names = cancer_df[1].unique()

# Plot the data points with colors based on target values
plt.figure(figsize=(8, 6))
colors = ['r', 'g']
for target, color in zip(target_names, colors):
    indicesToKeep = final_df[1] == target
    plt.scatter(final_df.loc[indicesToKeep, 'PC1'],
                final_df.loc[indicesToKeep, 'PC2'],
                c=color,
                s=50)

# Add legend with target names
plt.legend(target_names)

# Add labels and title
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA of Breast Cancer Dataset')

# Show the plot
plt.grid()
plt.show()

