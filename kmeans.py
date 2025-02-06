from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, D, K):
        self._data = D
        self._m = D.shape[1]
        self._n = D.shape[0]

        #Getting the indices of the centroids        
        centroid_indices = kmeans_pp(D, K)
        self._centroid_indices = centroid_indices

        #Getting the distance of each point di from the defined centroids
        centroid_distances = kmeans_distances(D, centroid_indices)
        self._centroid_distances = centroid_distances

        #Determining the cluster each point di belongs to. It is the centroid it is closest to
        labels = kmeans_classif(centroid_distances)
        self._labels = labels

        if (D.shape[1] > 3):
            pca = PCA(n_components=3)
            self._reduced_data = pca.fit_transform(D)
        else:
            self._reduced_data = D     

    def get_data(self):
        return self._data
    
    def get_centroid_indices(self):
        return self._centroid_indices
    
    def get_centroid_distances(self):
        return self._centroid_distances
    
    def get_labels(self):
        return self._labels
    
    def get_reduced_data(self):
        return self._reduced_data

    def get_n(self):
        return self._n
    
    def get_m(self):
        return self._m

    def plot_clustered_data(self):
        labels = self.get_labels()

        #Getting the reduced data to plot it
        D = self.get_reduced_data()
        dimensions = D.shape[1] 
            
        if dimensions == 3:
            fig = plt.figure(figsize = (10, 7))
            ax = fig.add_subplot(111, projection='3d')

            scatter = ax.scatter(D[:, 0], D[:, 1], D[:, 2], c = labels, cmap = 'viridis', s = 50)

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            fig.colorbar(scatter, label = "Cluster Label")

            plt.title("3D Scatter Plot of K-Means Clustering")
        
        if dimensions == 2:
            plt.scatter(D[:, 0], D[:, 1], c = labels)
            plt.xlabel("X")
            plt.ylabel("Y")  

            plt.title("2D Scatter Plot of K-Means Clustering")

        plt.show()

    def plot_raw_data(self):
        D = self.get_reduced_data()
        dimensions = D.shape[1] 

        if dimensions == 3:
            fig = plt.figure(figsize = (10, 7))
            ax = fig.add_subplot(111, projection = '3d')
            
            scatter = ax.scatter(D[:, 0], D[:, 1], D[:, 2], s = 50)
            
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            plt.title("3D Scatter Plot of Raw Data")

        if dimensions == 2:
            plt.scatter(D[:, 0], D[:, 1])
            plt.xlabel("X")
            plt.ylabel("Y")

            plt.title("2D Scatter Plot of Raw Data")

        plt.show()

def kmeans_pp(D, K):
    """
    This is an implementation of the KMeans++ Algorithm proposed by Arthur and Vassilvitskii in "kmeans++: The Advantages of Careful Seeding"

    Parameters: D -> A set of data points
                K -> The number of clusters
    Returns: centroids -> A list that contains the indices of the centroids 
    """
    n = D.shape[0]

    centroids = []

    #Setting the initial centroid as the point closest to the origin
    distance_from_origin = np.linalg.norm(D, axis=1)
    initial_centroid_index = np.argmin(distance_from_origin)
    centroids.append(initial_centroid_index)

    #Finding the K-1 points with maximum distance from the centroids
    for i in range(1, K):
        min_distances = np.array([min(np.linalg.norm(D[j] - D[centroid_index]) for centroid_index in centroids) for j in range(n)])

        min_distances[centroids] = 0

        next_centroid_index = np.argmax(min_distances)
        centroids.append(next_centroid_index)

    return centroids

def kmeans_distances(D, centroid_indices):
    """
    This function calculates the Euclidean distance between each point and the K centroids.

    Parameters: D -> The data points
                centroids -> An array containing the indices of the clusters 
    Returns: centroid_distances (n, K) -> An array containing the distances of each point from the centroids 
    
    """
    n = D.shape[0]
    K = len(centroid_indices) 

    print("Centroid Indices:\t", centroid_indices)
    centroid_distances = np.zeros((n, K))

    for i in range(n):
        di = D[i] 
        distances = []

        for centroid_index in centroid_indices: 
            centroid = D[centroid_index] 
            distances.append(np.linalg.norm(centroid - di))

        centroid_distances[i, :] = distances

    return centroid_distances

def kmeans_classif(centroid_distances):
    """
    This function classifies the centroids to their appropriate clusters. It determines this by finding the index of the smallest Euclidean distance in the centroid_distances array.

    Parameters: centroid_distances (n, K) -> An array containing the Euclidean distances of each point and the centroids
    Returns: data_classification (n, 1) -> A 1D Array that will contain the cluster each point belongs to 
    """
    n = centroid_distances.shape[0]
    data_classification = np.zeros(n)
    
    for i in range(n):
        data_classification[i] = np.argmin(centroid_distances[i]) + 1

    return data_classification

if __name__ == "__main__":
   np.random.seed(50)
   n = 4000
  
   D = np.random.rand(n, 9)

   test = KMeans(D, 4)
   test.plot_clustered_data()