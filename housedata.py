import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\\Users\\Mani Krishna Karri\\Downloads\\Makaan Dataset.csv")
valid_keywords = ['New', 'Under', 'Ready', 'year', 'years', 'old']
is_valid_age = df['Age of property'].astype(str).str.contains('|'.join(valid_keywords), case=False)
df.loc[~is_valid_age, 'Age of property'] = np.nan
#fun to convert age
def convertage(age_str):
    if pd.isna(age_str):
        return np.nan

    age_str = str(age_str).lower().strip()


    if 'new' in age_str:
        return 0.0

  
    if 'year' in age_str and '-' not in age_str:
        for part in age_str.split():
            if part.isdigit():
                return float(part)

    if '-' in age_str:
        nums = [int(s) for s in age_str.replace('-', ' - ').split() if s.isdigit()]
        if len(nums) == 2:
            return sum(nums) / 2.0

    return np.nan

# Apply the function
df['Age of property'] = df['Age of property'].apply(convertage)
df['Age of property'] =df['Age of property'].fillna(df['Age of property'].mean())
df.drop(['Builder Name','City','Unnamed: 0'],axis=1,inplace=True) 
status_map = {
    'Under Construction': 4,
    'New': 3,
    'Ready to move': 2,
    'Resale': 1
}
df['Construction Status'] = df['Construction Status'].map(status_map)
isvalid_type = df['Property Type'].astype(str).str.contains('BHK',case=False)
df.loc[~isvalid_type,'Property Type'] = np.nan
def convertptype(type_str):
    if pd.isna(type_str):
        return np.nan
    type_str = str(type_str).lower().strip()
    for part in type_str.split():
        if part.isdigit():
            return int(part)
df['Property Type'] = df['Property Type'].apply(convertptype)
df['Property Type'] =df['Property Type'].fillna(df['Property Type'].median())
df.rename(columns={
    'Property Type': 'Property_Type',
    'Price in Crores': 'Price_in_Crores',
    'Area in Sqft': 'Area_in_Sqft',
    'Construction Status': 'Construction_Status',
    'Age of property': 'Age_of_property'
}, inplace=True)
locality_grouped = df.groupby('Locality').agg({
    'Price_in_Crores': 'mean',
    'Area_in_Sqft': 'mean'
}).dropna()
locality_grouped_new = locality_grouped.reset_index()

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
X = locality_grouped_new[['Price_in_Crores', 'Area_in_Sqft']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  

kmeans = KMeans(n_clusters=4,random_state=42)
kmeans.fit(X_scaled)
locality_grouped_new['Cluster'] = kmeans.labels_
#map to cluster
locality_cluster_map = locality_grouped_new.set_index('Locality')['Cluster'].to_dict()
df['Locality_Cluster'] = df['Locality'].map(locality_cluster_map)
df['Price_in_Crores'] = np.log1p(df['Price_in_Crores'])


x = df.drop(['Locality','Price_in_Crores'],axis=1)
y = df['Price_in_Crores']
X_train,X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# from sklearn.linear_model import LinearRegression
# reg = LinearRegression()
# reg.fit(X_train,y_train)
# y_pred = reg.predict(X_test)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)


import joblib

joblib.dump(rf,'house_predict_rf.pkl')
joblib.dump(locality_cluster_map, "locality_cluster_map.pkl")










