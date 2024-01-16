import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics

# Reading house price data and latitude_longitude data
house_pricing_data = pd.read_csv("school\Introduction to Machine Learning\Week07_exercise\data\house_pricing_data.csv")
latitude_longitude = pd.read_csv("school\Introduction to Machine Learning\Week07_exercise\data\latitude_longitude.csv")
# print(house_pricing_data.columns)
# print(latitude_longitude.columns)


house_machi_code = pd.DataFrame(house_pricing_data['市区町村コード'])
latitude_longitude_reference = latitude_longitude[['コード', '緯度', '経度']]

house_machi_code.dropna(inplace = True)
latitude_longitude_reference.dropna(inplace = True)

datacheck_house_machi_code = pd.DataFrame(house_machi_code).map(lambda x : len(str(x)))
datacheck_latitude_longitude_reference = pd.DataFrame(latitude_longitude_reference['コード']).map(lambda x : len(str(x)))

print(datacheck_house_machi_code.value_counts())
print(datacheck_latitude_longitude_reference.value_counts())

code2lng = {}
code2lat = {}
for i in range(len(latitude_longitude_reference)):
    code = str(latitude_longitude_reference['コード'].iloc[i])
    if (len(code) == 5):
        code = code[0:4]
    else:
        code = code[0:5]
    code2lng.update({int(code) : latitude_longitude_reference["経度"].iloc[i]})
    code2lat.update({int(code) : latitude_longitude_reference["緯度"].iloc[i]}) 

house_machi_code['latitude'] = house_machi_code['市区町村コード'].map(code2lat)
house_machi_code['longitude'] = house_machi_code['市区町村コード'].map(code2lng)

house_location = []
for i in range(len(house_machi_code)):
    lat = house_machi_code.iloc[i, 1]
    long = house_machi_code.iloc[i, 2]
    house_location.append((lat, long))

print(len(house_location))

# plot the house positions
house_machi_code.plot(x = 'latitude', y = 'longitude', kind = 'scatter', label = 'House location', marker = '+', color = 'black')
plt.title("Scatterplot of house locations")
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.show()

location_array = house_machi_code[['latitude', 'longitude']].to_numpy()

kmeans = KMeans(n_clusters=10, random_state = 42)
kmeans.fit(location_array)

centers = kmeans.cluster_centers_
print(centers)

predicted_labels = kmeans.labels_
print(len(predicted_labels))

merged_array = np.concatenate((location_array, predicted_labels.reshape(-1, 1)), axis =1)

print(merged_array)

labels = np.unique(merged_array[:, 2])
split_house_array = [merged_array[merged_array[:,2] == label] for label in labels]
label_count = 0
for array in split_house_array:
    plt.scatter(x = array[:,0], y = array[:,1], label = 'house cluster ' + str(label_count), marker = '+')
    label_count += 1
plt.scatter(x = centers[:,0], y = centers[:,1], label = 'house location', marker = '+', color = 'red')
plt.title("Cluster result of house locations")
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.show()

## Evaluation
silhousette = metrics.silhouette_score(location_array, predicted_labels)
print("Silhousette Score: ", silhousette)

calinski = metrics.calinski_harabasz_score(location_array, predicted_labels)
print("Calinski-Harabasz: ", calinski)

Davies = metrics.davies_bouldin_score(location_array, predicted_labels)
print("Davies-Bouldin: ", Davies)

