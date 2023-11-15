install.packages("cluster")

library(tidyverse)
library(ggplot2)
library(cluster)



data(ruspini, package = "cluster")
ruspini <- as_tibble(ruspini) %>% sample_frac()
ruspini
ggplot(ruspini, aes(x = x, y = y)) + geom_point()
summary(ruspini)


scale_numeric <- function(x) x %>% mutate_if(is.numeric, function(y) as.vector(scale(y)))
ruspini_scaled <- ruspini %>% scale_numeric()
ggplot(ruspini_scaled, aes(x = x, y = y)) + geom_point()
summary(ruspini_scaled)

############################K-means##############################
km <- kmeans(ruspini_scaled, centers = 4, nstart = 10)
km

ruspini_clustered <- ruspini_scaled %>% add_column(cluster = factor(km$cluster))
ruspini_clustered
ggplot(ruspini_clustered, aes(x = x, y = y, color = cluster)) + geom_point()

centroids <- as_tibble(km$centers, rownames = "cluster")
centroids
ggplot(ruspini_clustered, aes(x = x, y = y, color = cluster)) + geom_point() + geom_point(data = centroids, aes(x = x, y = y, color = cluster), shape = 3, size = 10)

install.packages("factoextra")
library(factoextra)
fviz_cluster(km, data = ruspini_scaled, centroids = TRUE, repel = TRUE, ellipse.type = "norm")

ggplot(pivot_longer(centroids, cols = c(x, y), names_to = "feature"),
  aes(x = value, y = feature, fill = cluster)) +
  geom_bar(stat = "identity") +
  facet_grid(rows = vars(cluster))

cluster1 <- ruspini_clustered %>% filter(cluster == 1)
cluster1
summary(cluster1)
ggplot(cluster1, aes(x = x, y = y)) + geom_point() + coord_cartesian(xlim = c(-2, 2), ylim = c(-2, 2))

fviz_cluster(kmeans(ruspini_scaled, centers = 8), data = ruspini_scaled,
  centroids = TRUE,  geom = "point", ellipse.type = "norm")



############################Hierarchical Clustering##############################

d <- dist(ruspini_scaled)
hc <- hclust(d, method = "complete")
plot(hc)
fviz_dend(hc, k = 4)

clusters <- cutree(hc, k = 4)
cluster_complete <- ruspini_scaled %>%
  add_column(cluster = factor(clusters))
cluster_complete

ggplot(cluster_complete, aes(x, y, color = cluster)) +
  geom_point()


fviz_cluster(list(data = ruspini_scaled, cluster = cutree(hc, k = 8)), geom = "point")


hc_single <- hclust(d, method = "single")
fviz_dend(hc_single, k = 4)

fviz_cluster(list(data = ruspini_scaled, cluster = cutree(hc_single, k = 4)), geom = "point")




############################Internal Cluster Validation##############################
install.packages("fpc")
library(fpc) 
cluster.stats(d, km$cluster)


library(cluster)
plot(silhouette(km$cluster, d))
fviz_silhouette(silhouette(km$cluster, d))


############################Find Optimal Number of Clusters for k-means##############################
ggplot(ruspini_scaled, aes(x, y)) + geom_point()
set.seed(1234)
ks <- 2:10

WCSS <- sapply(ks, FUN = function(k) {
  kmeans(ruspini_scaled, centers = k, nstart = 5)$tot.withinss
  })

ggplot(as_tibble(ks, WCSS), aes(ks, WCSS)) + geom_line() +
  geom_vline(xintercept = 4, color = "red", linetype = 2)



ASW <- sapply(ks, FUN=function(k) {
  fpc::cluster.stats(d, kmeans(ruspini_scaled, centers=k, nstart = 5)$cluster)$avg.silwidth
  })

best_k <- ks[which.max(ASW)]
best_k

ggplot(as_tibble(ks, ASW), aes(ks, ASW)) + geom_line() +
  geom_vline(xintercept = best_k, color = "red", linetype = 2)




############################Visualize the Distance Matrix##############################

ggplot(ruspini_scaled, aes(x, y, color = factor(km$cluster))) + geom_point()

d <- dist(ruspini_scaled)
as.matrix(d)[1:5, 1:5]

install.packages("seriation ")
library(seriation)
pimage(d, col = bluered(100))

pimage(d, order=order(km$cluster), col = bluered(100))

dissplot(d, labels = km$cluster, options=list(main="k-means with k=4"))

dissplot(d, labels = kmeans(ruspini_scaled, centers = 3)$cluster, col = bluered(100))

dissplot(d, labels = kmeans(ruspini_scaled, centers = 9)$cluster, col = bluered(100))

fviz_dist(d)