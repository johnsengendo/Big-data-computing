# Big-data-computing
## Clustering of data points
In this project I cluster points based on their similarities which is their closest distance , different clusters are created in the data and the center of each clustar is selected among the data points.
The task of the project was to find the number of clusters [ selected between 8 - 12] in the dataset that give the best silhouette value.
I run the program by inputting the number of clusters needed and centers are selected from the data.
* I returns the sum of squared euclidean distances between the pair of points. That is, a point and its closest center.
* To evaluate similarities among points, I use the KMean clustering algorithm where distace computation among points are computed.  
* Points that are close to a   particular center are put in a similar cluster.
### Clustering illustration using silhouette <br>
![BER](clustering.JPG)

* Silhouette coefficient was used to evaluate how good my clustering was. 
* A value near +1 indicates that the sample points are far away from the neighboring clusters and so points are well clusterd. 
* A value of 0 indicates that the sample points are on or very close to the decision boundary between two neighboring clusters and negative values indicates that the sample points might have been assigned to the wrong cluster.
