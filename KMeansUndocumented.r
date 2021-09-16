# R code for K-Means Clustering
## Dr. Clif Baldwin
## Autumn 2017; Updated February 26, 2019
# Uses one large function and does not vectorize, but it works and is "easier" to read than a vectorized attempt

# K-Means Clustering Algorithm pseudocode
# 1. Create k points for starting centroids 
# 2. While any point has changed cluster assignment
# 2a. For every point in our dataset:
# 2ai. For every centroid:
# 2ai1.  Calculate the distance between the centroid and point
# 2aii. Assign the point to the cluster with the lowest distance
# 2b. For every cluster calculate the mean of the points in that cluster
# 2bi. Assign the centroid to the mean

#Edited by: Nicholas Paisley

library(ggplot2) #the library to be loaded in
library(standardize)
library(tidyverse)

#data1 <- read.csv("KMeansData_Group1.csv", header = FALSE)

my_kmeans=function(data1,K=5,stop_crit=10e-3) #set a variable up using the function definition. All the statements in the in the function are arguments/values that are going to be used within the function body.
#data1 is the dataset, K is the number of centroids that is needed in the dataset, stop_crit stops the algorithm when close enough to the minimum (in this case 0.01)
  {
  #INITIALISATION OF CLUSTERS/CENTROIDS
  centroids=data1[sample.int(nrow(data1),K),] #Centroid initialization. Establishes the number of K (initial centroids) within the dataset(s). sample_int, our n is the number of rows in our dataset, size is our defined K variable, prob is null.
  #sample.int(n,size,prob) 
    #n = a positive number, the number of items to chose from
    #size = a non-negative number of the items to choose
    #prob = a vector of probability weights for obtaining the elements of the vector being sampled.
  current_stop_crit=1000 #Stopping criteria initialization - 
  cluster=rep(0,nrow(data1)) #rep(value, number_of_times)
  converged = FALSE #The function has not converged. When the centroids no longer change and the cluster memberships stabilize.
  it=1 #iteration that starts at 1
  
  while(current_stop_crit>=stop_crit && ! converged) #arguments both must be met for the while to activate. 
  {
    it=it+1 #iterations that go up by 1 if neither of the two if statements are true. If one of the IF statements is true, then it moves on. 
    if (current_stop_crit<=stop_crit) converged = TRUE
    if (it > 20) { converged = TRUE } #if iterations are above 20, convergence is forced to be true.

    old_centroids=centroids #initial centriods are relabed as old_centroids

    for (i in 1:nrow(data1)) #iterating over data1 observations individually.
    {
      min_dist=10e10 #setting a high minimum distance. Find the distance between each centroid between each centroid. The closet centroid is where the data point is going to be assigned to. 
      for (centroid in 1:nrow(centroids)) #Iterating over centroids for the centroid in the set of initial centroids(old_centroids)
      {
        distance_to_centroid=sum((centroids[centroid,]-data1[i,])^2) #Computing the L2 distance. Computing minimum euclidean distance between datapoint and the nearest centroid
        if (distance_to_centroid<=min_dist) #If this centroid is the closest centroid to the point by distance_to_centroid <= minimum distance
        {
          #Then the data point is assigned to this centroid/cluster
          cluster[i]=centroid 
          min_dist=distance_to_centroid #distance_to_centroid is relabelled as the new minimum distance
        }
      }
    }
    for (i in 1:nrow(centroids)) #for each of the centroids in old_centroids
    {
      centroids[i,]=apply(data1[cluster==i,],2,mean) #calculating the mean of each cluster group to determine the new centroids in the center of the clusters
    }
    current_stop_crit=mean((old_centroids-centroids)^2) #difference of old_centroids(initial) and mean of clusters(new centroids). Establishing a smaller current_stop_crit to be used back in the beggining of the while statement
  }                                                     #If the the current_Stop_crit less than or equal to (<=) stop_crit, the model converged and the iterations stopped. 
  
  centroids = data.frame(centroids, cluster=1:K) #creates a new centroid variable for a dataframe and with the initial centroids and there cluster...
  return(list(data1=data.frame(data1,cluster),centroids=centroids)) #When alogrithm is finished 
}

###### For just a given K ######

DF <- read.csv("KMeansData_Group1.csv", header = FALSE) #read in data as DF
ggplot(DF,aes(x=V1,y=V2))+geom_point() #Graph 1 - Simple black dot graph. Made to see if data was imported correctly.

DF <- as.matrix(DF) #Converts data table into a matrix. Used to help plot algorithm from scratch code. 

k = 4L #K value where L is stating that the variable is an integer
res=my_kmeans(DF,K=k) 

#Setting up parameters to plot the clusters and 
res$data1$isCentroid=FALSE #Defining regular data points 
res$centroids$isCentroid=TRUE #Defining centroids of clusters
data_plot=rbind(res$centroids,res$data1) #combining datapoints and centroids into one dataframe. Plotting below
ggplot(data_plot,aes(x=V1,y=V2,color=as.factor(cluster),size=isCentroid,alpha=isCentroid))+geom_point() #Graph 2 - Colorful

# Compute the total within-cluster sum of squares (wss)
wss = 0.0 #initial wss variable with no calculation.

for (i in 1:k) { #iterating through the number of clusters starting with the first one.
  for (j in 1:nrow(res$data1)) { #For each row in the res$data1 (the data used to create the first visual)
    if(res$data1$cluster[j] == res$centroids$cluster[i]) { #If the assigned cluster in the data matches the centroid cluster, the code moves forward to compute the wss
      wss = wss + (res$data1$V1[j] - res$centroids$V1[i])^2 + (res$data1$V2[j] - res$centroids$V2[i])^2 # wss (initial 0.0, then the sum of cluster 1 ,2 ...etc ) + (x-value of data point - x-value of centroid point)^2 + (y-value of data point - y-value of centroid point)^2 , then iterates through other cluster while storing completed cluster calculation in wss variable.
      #Calculating the Euclidean distances of each point to the centroids of each cluster    
      }
  }
}
rm(i, j) #Removes the variables so they will not be stored in memory.

#normalize <- function(x) { (x - min(x)) / (max(x) - min(x))}

#data1_norm <- as.data.frame(lapply(data1[,c(1,2)], normalize))

ks <- 2:10 # different K's 
# I do not want K=1 because that would mean everything is part of one group,

# Initialize a vector to hold each K's Total Within-Sum of Squares
kss = vector(mode = "numeric", length = length(ks))

# Compute the K Means Clusters for clusters k = 2 to 25
# kmeans() is automatically loaded with R
for(i in ks) {
  kdata1 <- kmeans(res$data1[,c(1,2)], centers = i)
  kss[i-1] = kdata1$tot.withinss # Save each Total Sum of Squares to the vector
} # Now the K Means has been computed for K = 2, 3, 4, ..., 25


# Print the Elbow Curve
# The x-axis are the ks (i.e. K=2, 3, 4, 5, ..., 25)
# The y-axis are the Total Sum of Square errors
plot(ks, kss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares",
     main="The Elbow Curve of Data1 Set",
     pch=20, cex=2)

# Compare the output of the R function kmeans()
test <- kmeans(DF, k, nstart=1)
plot(DF, col=test$cluster, main="Using kmeans()") #Grpah 3 - shows clusters by color. This graph is using the real K-Means Function (kmeans())

###Normalize the data###
  
#DF_norm <- standardize(DF)

normalize <- function(x) { (x - min(x)) / (max(x) - min(x))} #normalizes the data 

is.standardized(DF) #Standardize data with given functions for computing center and scale.
DF_scaled <- scale(DF) #scales the columns of a numeric matrix
is.standaridized(DF_scaled) 

glimpse(DF_scaled)

DF_norm <- as.data.frame(DF_scaled, normalize)  #Creating a dataframe and normalizing the new DF_scaled

glimpse(DF_norm)

ggplot(DF_norm, aes(x=V1,y=V2)) + geom_point() + labs(title = 'Normalized Data') #plotting the "Normalized Data"
