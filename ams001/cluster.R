install.packages("cluster")
install.packages("tidyverse")
install.packages("ggplot2")

library(tidyverse)
library(ggplot2)
library(cluster)


heartdata= read.csv("dataset/heart.csv")
class(heartdata)

#Exploring data with 0,1 values as they won't be of much help in making the clusters
heartdata_c = heartdata[,-which(colnames(heartdata) %in% c("sex","fbs","exng","output"))]

# Standardize the dataset and apply K-means clustering
heartdata_scale = scale(heartdata_c)
summary(heartdata_scale)

km <- kmeans(heartdata_scale, centers = 2, nstart = 10)
km

heartdata_clustered <- as_tibble(heartdata_scale)  %>% add_column(cluster = factor(km$cluster))
heartdata_clustered
ggplot(heartdata_clustered, aes(x = heartdata_clustered$age, y = heartdata_clustered$thalachh, color = cluster)) + geom_point()
#actual output
ggplot(heartdata, aes(x = heartdata$age, y = heartdata$thalachh, color = heartdata$output)) + geom_point()


#calculates the accuracy of the prediction
accuracy = ifelse(heartdata_clustered$cluster == heartdata$output+1, 'Y','N')
acc_rate = sum(accuracy == 'Y')/length(accuracy)
acc_rate

