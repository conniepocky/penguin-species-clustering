import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data/penguins.csv")

# clean data

data = data.dropna()
df = pd.get_dummies(data).drop("sex_.", axis=1)

# pre processing

scaler = StandardScaler()
X = scaler.fit_transform(df)
penguins_preprocessed = pd.DataFrame(data=X, columns=df.columns)

# pca is a dimensionality reduction technique, used for visualization

pca = PCA(n_components=None)
dfx_pca = pca.fit(penguins_preprocessed)
dfx_pca.explained_variance_ratio_
n_components = sum(dfx_pca.explained_variance_ratio_ > 0.1)
pca = PCA(n_components=n_components)
penguins_PCA = pca.fit_transform(penguins_preprocessed)

# elbow method to find the optimal number of clusters

inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(penguins_PCA)
    inertia.append(kmeans.inertia_)
plt.plot(range(1, 10), inertia, marker="o")
plt.xlabel("Number of clusters")
plt.show()

k = 5

# kmeans clustering

kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(penguins_PCA)
clusters = kmeans.predict(penguins_PCA)

# plot the clusters

plt.scatter(penguins_PCA[:, 0], penguins_PCA[:, 1], c=clusters, s=50, cmap="viridis")
plt.show()
