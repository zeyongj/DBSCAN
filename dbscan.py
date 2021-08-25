import numpy as np
import pandas as pd
import math
import queue
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors # Only for part c.1

# The following codes are for part b, DBSCAN. The part of returning a file is in the main function.
UNASSIGNED = -2
NOISE = -1

class DBSCAN():
    def __init__(self, epsilon, MinPts):
        self.epsilon = epsilon
        self.MinPts = MinPts

    def distance(self, x, y):
        ans = math.sqrt(np.power(x-y, 2).sum())
        return ans

    def neighbours(self, example, objectID):
        objects = []
        for i in range(len(example)):
            dist = self.distance(example[i, :], example[objectID, :])            
            if dist < self.epsilon:
                objects.append(i)
        return np.asarray(objects)

    def clustering(self, example, clusterLabel, objectID, clusterID):
        objects = self.neighbours(example, objectID)
        objects = objects.tolist()
        myQueue = queue.Queue()
        lengthObj = len(objects)
        
        if lengthObj < self.MinPts:
            clusterLabel[objectID] = NOISE
            return False
        else:
            clusterLabel[objectID] = clusterID
        
        for object in objects:
            if clusterLabel[object] == UNASSIGNED:
                myQueue.put(object)
                clusterLabel[object] = clusterID
        while not myQueue.empty():
            neighboursRemaining = self.neighbours(example, myQueue.get())
            lengthNbo = len(neighboursRemaining)
            if lengthNbo >= self.MinPts:
                for i in range(lengthNbo):
                    resultObject = neighboursRemaining[i]
                    if clusterLabel[resultObject] == UNASSIGNED:
                        myQueue.put(resultObject)
                        clusterLabel[resultObject] = clusterID
                    elif clusterLabel[clusterID] == NOISE:
                        clusterLabel[resultObject] = clusterID
        return True

    def fit(self, data):
        clusterID = 0
        length = len(data)
        clusterLabel = [UNASSIGNED] * length
        for objectID in range(length):
            if clusterLabel[objectID] == UNASSIGNED:
                if self.clustering(data, clusterLabel, objectID, clusterID):
                    clusterID += 1
        return np.asarray(clusterLabel)

def main():
    # This part is for part a, preprocessing.
    print('......Reading Data......\n')
    data_file = "houshold2007.csv"
    print('......Reading Completed......\n')
    print('......Preprocessing......\n')
    data = pd.read_csv(data_file, sep = ",",low_memory = False)
    data = data[data['Date'].apply(lambda x: x.endswith('/1/2007'))] # Using January's data.
    data = data.iloc[:, [2,3,4,5]] # Using the data from 3rd to 6th column
    data = data.replace('?', np.nan)
    data = data.dropna()
    data = data.astype(float)
    data = data.apply(lambda x : (x-np.min(x))/(np.max(x)-np.min(x))) # Using min-max scaling.
    print('Further Explanation And Analysis of Preprocessing Is in The Report.\n')
    print('......Preprocessing Completed......\n')
    
    # This part is for part b, implementing DBSCAN and returning a file of cluster labels.
    print('......Implementing DBSCAN......\n')
    data = data.values
    clusterLabel = DBSCAN(epsilon = 0.030, MinPts = 90).fit(data)
    labelSeries = pd.Series(clusterLabel)
    cluster = list(labelSeries.value_counts().index) 
    frequency = list(labelSeries.value_counts())
    print('Number of cluster labels is %d.' % (len(cluster)))
    print('Cluster labels are the following. ATTENTION: NOISE is -1, cluster labels start from 0.')
    print(cluster)
    print('Frequency of each cluster label is the following.')
    print(frequency)
    print('Further Explanation And Analysis of Histogram Is in The Report.\n')
    print('......Generating Cluster Label File......\n')
    with open("Output of Cluster Labels.txt","w") as f:
        f.write('Cluster labels are the following. ATTENTION: NOISE is -1, cluster labels start from 0.\n'+ str(cluster))
    print('......Cluster Label File Generated......\n')
    print('......Implementation Completed......\n')

    # This part is for part c.1, k distance diagram.
    # The following codes are modified and inspired by the website: https://medium.com/@tarammullin/dbscan-parameter-estimation-ff8330e3a3bd.
    # The following codes will use sklearn.neighbors.
    print('......Generating Figure 1: K Distance Diagram......\n')
    dimention = 4
    n_neighbors = 2 * dimention
    neighbors = NearestNeighbors(n_neighbors=8)
    neighbors_fit = neighbors.fit(data)
    distances, indices = neighbors_fit.kneighbors(data)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances)
    plt.xlabel("Number of Records")
    plt.ylabel("Epsilon")
    plt.title("Figure 1: K Distance Diagram")
    plt.savefig("Figure 1_K Distance Diagram.png")
    print('Further Explanation And Analysis of This Diagram Is in The Report.\n')
    plt.show()
    print('......Figure 1: K Distance Diagram Generated......\n')

    # This part is for part c.2, statistics of clustering.
    print('......Generating Figure 2: Histogram of Clusters......\n')
    plt.bar(cluster, frequency, width=0.5)
    plt.xlabel("Clusters")
    plt.ylabel("Frequency")
    plt.title("Figure 2: Histogram of Clusters")
    plt.savefig("Figure 2_Histogram of Clusters.png")
    print('Further Explanation And Analysis of This Histogram Is in The Report.\n')
    plt.show()
    print('......Figure 2: Histogram of Clusters Generated......\n')

if __name__ == '__main__':
    print('......This Is The Beginning of Assignment 2......\n')
    main()
    print('......This Is The End of Assignment 2......\n')
    
    
    

