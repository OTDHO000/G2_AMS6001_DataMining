### Package Installation (ONLY when it is not installed before)
#install.packages("tidyverse")
#install.packages("ggplot2")
#install.packages("GGally")
#install.packages("mlbench")
#install.packages("caret")
#install.packages("rpart")
#install.packages("leaps")


### Load packages and file
library(tidyverse)
library(ggplot2)
library(GGally)
library(mlbench)
library(caret)
library(rpart)
library(leaps)

setwd("~/Projects/HSUHK/AMS6001/Proj/Data/") #adjust to file location
heart <- read.csv("heart.csv")


## I. Data Preparation

### Cleaning missing data
heart_data <- heart %>% drop_na()
summary(heart_data)
### Separate categorical and numerical indices
categories_var <- c("sex","cp","fbs","slp","exng","restecg","caa","thall","output")
numerical_var <- setdiff(names(heart_data), categories_var)


## II. EDA

### 2.1 Univariate distribution
# Boxplot of all variables
# Convert the dataframe to the long format
data_long <- tidyr::gather(heart_data[numerical_var], key = "Variable", value = "Value")

# Create separate boxplots for each variable and display them together
ggplot(data_long, aes(x = "", y = Value)) +
  geom_boxplot() +
  facet_wrap(~ Variable, scales = "free") +
  labs(x = "", y = "Value") +
  ggtitle("Boxplots of Numerical Variables")

# Set up the grid of subplots
par(mar = c(2, 4, 2, 1))
par(mfrow = c(4, 4))  # Creates a 4x4 grid of subplots

# Loop through each variable and plot its distribution
for (i in 1:ncol(heart_data)) {
  hist(heart_data[, i], main = colnames(heart_data)[i], col = "lightblue")
}

# Reset the plotting parameters to default
par(mfrow = c(1, 1))

### 2.2 Bivariate relationship between numerical attributes
# Select a subset of variables for the scatter plot matrix
# Create the scatter plot matrix
selected_vars <- c("age", "chol", "trtbps", "thalachh", "oldpeak")
pairs(heart_data[selected_vars], pch = 19, main = "Scatterplot matrix of numerical variables")
# 
selected_vars <- c("age", "chol", "trtbps", "thalachh", "oldpeak", "output")
heart_tibble <- as_tibble(heart_data[selected_vars])
heart_tibble$output <- as.factor(heart_tibble$output)
ggpairs(heart_tibble, title = "Correlation between numerical variables with respect to target", aes(color = output))

### 2.3 Correlation between predictors and target variable
# One-Hot Encoding for categorical variables
encoded_data <- model.matrix(~., data = heart_data[, -14])

# Merge one-hot encoded data and target variable
processed_data <- cbind(encoded_data, target = heart_data$output)

# Calculate correlation between numerical variables and target variable
correlation_matrix <- cor(heart_data[,-14], heart_data$output)
correlation_matrix

# Sort the variables with the absolute value of correlation in descending order
sorted_correlation_matrix <- correlation_matrix[order(abs(correlation_matrix[,1]), decreasing = TRUE),]
sorted_correlation_matrix
barplot(sorted_correlation_matrix, las=2, ylab = "Correlation", main = "Correlation with Target Variable")

# Select variables that larger than 0.2
threshold <- 0.2
selected_variables <- which(abs(correlation_matrix) > threshold)

# Output
names(heart_data)[selected_variables]

### 2.4 Subset Selection with Backward Method
#### 2.4.1 Data Split and Factorization
set.seed(99999) #allow reproducing result
train_indices <- sample(1:nrow(heart_data), 0.85 * nrow(heart_data))

#training data and test data
train_data <- heart_data[train_indices, ]
test_data <- heart_data[-train_indices, ]

#an additional set of training data with categorical data factorized
train_data_k <- train_data
test_data_k <- test_data

train_data_k[,categories_var]<-lapply(train_data_k[,categories_var],as.factor)
test_data_k[,categories_var]<-lapply(test_data_k[,categories_var],as.factor)


#### 2.4.2 BIC Plot (with categorical data factorized)
result.all <-regsubsets(output ~ ., data = train_data_k, method = "backward", nvmax=22)
summary(result.all)
plot(summary(result.all)$bic, type="l", ylab="BIC")
##### Findings: 6 features (10 levels) -- sex, cp, trtbps, slp, caa, thall -- are important


## III. Model Building

### 3.1 Logistic Regression
#install.packages("ISLR") #if not installed before
#install.packages("glmnet") #if not installed before
library(ISLR)
library(glmnet)

#### 3.1.1 Trial using all predictors
#create model
model_LG <- glm(output ~ ., data = train_data_k, family = "binomial")
#evaluate model
predicted_output <- predict(model_LG, newdata = test_data_k, type = "response")
predicted_labels <- ifelse(predicted_output > 0.5, 1, 0)
#accuracy
accuracy <- mean(predicted_labels == test_data_k$output)
print(paste("Accuracy:", accuracy))
#kappa
confusion_matrix_LG <- confusionMatrix(factor(predicted_labels),factor(test_data_k$output))
kappa_LG <- confusion_matrix_LG$overall['Kappa']
print(paste("Kappa:", kappa_LG))

#### 3.1.2 Trial using 9 predictors (filtered by correlation with target)
#create model
model_LG9 <- glm(output ~ age + sex + exng + caa + cp + 
                   thalachh + slp + oldpeak + thall, data = train_data_k, family = "binomial")
#evaluate model
predicted_output_9 <- predict(model_LG9, newdata = test_data_k, type = "response")
predicted_labels_9 <- ifelse(predicted_output_9 > 0.5, 1, 0)
#accuracy
accuracy_9 <- mean(predicted_labels_9 == test_data_k$output)
print(paste("Accuracy:", accuracy_9))
#kappa
confusion_matrix_9 <- confusionMatrix(factor(predicted_labels_9),factor(test_data_k$output))
kappa_9 <- confusion_matrix_9$overall['Kappa']
print(paste("Kappa:", kappa_9))

#### 3.1.3 Trial using 6 predictors chosen by BIC minimum
model_bw <-glm(output ~  sex + cp + trtbps + slp + caa + thall, data = train_data_k, family = "binomial")
predicted_bw <- predict(model_bw, newdata = test_data_k, type = "response")
predictedbw_labels <- ifelse(predicted_bw > 0.5, 1, 0)
#accuracy
accuracy_bw <- mean(predictedbw_labels == test_data_k$output)
print(paste("Accuracy:", accuracy_bw))
#kappa
confusion_matrix_bw <- confusionMatrix(factor(predictedbw_labels),factor(test_data_k$output))
kappa_bw <- confusion_matrix_bw$overall['Kappa']
print(paste("Kappa:", kappa_bw))

#### 3.1.4 Comparison of accuracy
print(paste("Accuracy for all predictors:", accuracy, "; Kappa:", kappa_LG))
print(paste("Accuracy for 9 predictors:", accuracy_9, "; Kappa:", kappa_9))
print(paste("Accuracy for 6 predictors:", accuracy_bw, "; Kappa:", kappa_bw))

##### Conclusion of "3.1 Logistic Regression": Model with 6 predictors chosen by BIC test performs the best

### 3.1.5 Test whether the model is statistically significant by ANOVA
# Perform an ANOVA test
anova_result <- anova(model_bw)
anova_result

# Perform a chi-squared test to compare the deviance and null deviance
chi_squared <- model_bw$null.deviance - model_bw$deviance
df <- model_bw$df.null - model_bw$df.residual
p_value_chi <- pchisq(chi_squared, df, lower.tail = FALSE)

cat("Chi-squared test p-value:", p_value_chi)
##### Conclusion: The Chi-squared test on Deviance confirmed the model statistically

### 3.2 KNN (K-nearest neighbour) Method
#### 3.2.1 Trial with all predictors
# Set trainControl
ctrl <- trainControl(method = "cv", number = 10) # 10-fold Cross-validation

# Cross-validation using train() function
knn_model <- train_data_k %>% train(output ~ .,  # Using all predictors
                                    data = ., 
                                    preProcess = "scale", 
                                    method = "knn",   
                                    trControl = ctrl,  
                                    tuneGrid=data.frame(k = 1:16), 
                                    tuneLength = 10)   

knn_model
print(knn_model$results)

#### 3.2.2 Trial with six predictors filtered by BIC test (rerun)
#set trainControl
ctrl <- trainControl(method = "cv", number = 10) # 10-fold Cross-validation

# Cross-validation using train() function
knn_model6 <- train_data_k %>% train(output ~ sex+cp+oldpeak+slp+caa+thall,        # Using selected 6 variables
                                     data = ., 
                                     preProcess = "scale", 
                                     method = "knn",   
                                     trControl = ctrl,  
                                     tuneGrid=data.frame(k = 1:16), 
                                     tuneLength = 10)   

knn_model6
print(knn_model6$results)

#### 3.2.3 Trial with nine predictors (filtered by correlation)
#set trainControl
ctrl <- trainControl(method = "cv", number = 10) # 10-fold Cross-validation

# Cross-validation using train() function
knn_model9 <- train_data_k %>% train(output ~ age + sex + exng + caa + cp + thalachh + slp + oldpeak + thall,        # Using selected 9 variables
                                     data = ., 
                                     preProcess = "scale", 
                                     method = "knn",   
                                     trControl = ctrl,  
                                     tuneGrid=data.frame(k = 1:16), 
                                     tuneLength = 10)   

knn_model9
print(knn_model9$results)

#### 3.2.4 Compare the Accuracy and Kappa of the two models
# Compare the Accuracy and Kappa of the two models
resamps <- resamples(list(
  KNN = knn_model,
  KNN6 = knn_model6,
  KNN9 = knn_model9
))
resamps
library(lattice)
bwplot(resamps, layout = c(3, 1))


### 3.3 Decision Tree Method
#### 3.3.1 Trial with all predictors
#install.packages("rpart") #if not installed before
library(rpart)
library(rpart.plot)
train_data$output <- as.factor(train_data$output) # Factorize the target variable
train_index <- createFolds(train_data$output, k = 10)  # Create 10-fold data indcies for Cross-validation 
rpartFit1 <-  train(output ~ ., # Using all predictors
                    method = "rpart", # Using rpart decision tree method
                    data = train_data, 
                    tuneLength = 10,
                    trControl = trainControl(method = "cv", indexOut = 
                                               train_index))
rpartFit1
rpart.plot(rpartFit1$finalModel)
print(rpartFit1$results)

#### 3.3.2 Trial with 9 predictors
selected_vars <- c("age", "sex", "exng", "caa","cp","thalachh","slp", "oldpeak","thall", "output")
train_data_9 <- train_data[selected_vars]

train_index <- createFolds(train_data_9$output, k = 10)
rpartFit2 <-  train(output ~ .,
                    method = "rpart",
                    data = train_data_9,
                    tuneLength = 10,
                    trControl = trainControl(method = "cv", indexOut = 
                                               train_index))
rpartFit2
rpart.plot(rpartFit2$finalModel)
print(rpartFit2$results)

#### 3.3.3 Trial with 6 predictors suggested by backward subset selection
selected_vars <- c("sex", "cp", "thalachh","exng", "caa", "thall", "output")
train_data_6 <- train_data[selected_vars]

train_index <- createFolds(train_data_6$output, k = 10)
rpartFit3 <-  train(output ~ .,
                    method = "rpart",
                    data = train_data_6,
                    tuneLength = 10,
                    trControl = trainControl(method = "cv", indexOut = 
                                               train_index))
rpartFit3
rpart.plot(rpartFit3$finalModel)
print(rpartFit3$results)

#### 3.3.4 Trial with 6 predictors suggested by first tree
selected_vars <- c("age","sex", "cp", "thalachh", "caa", "thall", "output")
train_data_6_2 <- train_data[selected_vars]

train_index <- createFolds(train_data_6_2$output, k = 10)
rpartFit4 <-  train(output ~ .,
                    method = "rpart",
                    data = train_data_6_2,
                    tuneLength = 10,
                    trControl = trainControl(method = "cv", indexOut = 
                                               train_index))
rpartFit4
rpart.plot(rpartFit4$finalModel)
print(rpartFit4$results)

#### 3.3.5 Compare the Accuracy and Kappa of the four models
# Compare the Accuracy and Kappa of the two models
resamps <- resamples(list(
  DT  = rpartFit1,
  DT9 = rpartFit2,
  DT6 = rpartFit3,
  DT6_2 = rpartFit4
))
resamps
library(lattice)
bwplot(resamps, layout = c(3, 1))


### 3.4 Support Vector Machine (SVM) Method
#### 3.4.0 Further preparation
train_data_k_s <- train_data_k
test_data_k_s <- test_data_k
train_data_k_s[,numerical_var] <- scale(train_data_k_s[,numerical_var])
test_data_k_s[,numerical_var] <- scale(test_data_k_s[,numerical_var])

#### 3.4.1 Trial with all predictors
train_index <- createFolds(train_data_k_s$output, k = 10) 
svmFit <- train_data_k_s %>% train(output ~., # using all predictors
                                   method = "svmLinear",
                                   data = .,
                                   tuneLength = 10,
                                   trControl = trainControl(method = "cv", indexOut = train_index))
svmFit
svmFit$finalModel

#### 3.4.2 Trial with nine predictors
svmFit9 <- train_data_k_s %>% train(output ~ age + sex + exng + caa + cp + thalachh + slp + oldpeak + thall, # using nine selected predictors
                                    method = "svmLinear",
                                    data = .,
                                    tuneLength = 10,
                                    trControl = trainControl(method = "cv", indexOut = train_index))
svmFit9
svmFit9$finalModel

#### 3.4.3 Trial with six predictors
svmFit6 <- train_data_k_s %>% train(output ~ sex + cp + trtbps + slp + caa + thall, # using six selected predictors
                                    method = "svmLinear",
                                    data = .,
                                    tuneLength = 10,
                                    trControl = trainControl(method = "cv", indexOut = train_index))
svmFit6
svmFit6$finalModel

#### 3.4.4 Compare the Accuracy and Kappa of the three SVM models
# Compare the Accuracy and Kappa of the two models
resamps <- resamples(list(
  SVM = svmFit,
  SVM9 = svmFit9,
  SVM6 = svmFit6
))
resamps
bwplot(resamps, layout = c(3, 1))
##### Conclusion of "3.4 SVM": SVM taking 9 selected features into account performs better


### 3.5 Naive Bayes Method
#### 3.5.0 Further preparation
#install.packages("klaR") #if not installed before
library(klaR)

#### 3.5.1 Trial using all predictors BUT those problem-levels
train_data_k_n <- train_data_k
test_data_k_n <- test_data_k
col_to_be_dropped <- c("restecg", "caa", "thall")
col_remained <- setdiff(names(train_data_k_n), col_to_be_dropped)
train_data_k_n <- train_data_k_n[col_remained]
test_data_k_n <- test_data_k_n[col_remained]

train_data_k_n <- na.omit(train_data_k_n)
levels(train_data_k_n$output) <- c("NH", "H")
levels(test_data_k_n$output) <- c("NH", "H")
##### output '0' and '1' are renamed as 'NH' and 'H' respectively
##### (denoted "NO heart attack" and "Heart attack" respectively)
NBayesFit <- train_data_k_n %>% train(output ~ .,
                                      method = "nb",
                                      data = .,
                                      trControl = trainControl(
                                        method = "cv", # used for configuring resampling method: in this case cross validation 
                                        number = 10,
                                        index = createFolds(train_data_k_n$output, k = 10),
                                        classProbs = TRUE, 
                                        verboseIter = FALSE
                                      ),
                                      tuneLength = 10,
                                      tuneGrid = expand.grid(
                                        fL = 1:5,
                                        usekernel = c(TRUE, FALSE),
                                        adjust = 1:3
                                      )
)
##### our tuning grid consider three components:
##### [usekernel] we try both GUASSIAN and NON-GUASSIAN DISTRIBUTION of output
##### [fL] should be Laplace creation parameter to deal with zero probability in the middle, we let the programme try 1 to 5 and advise the best
##### [adjust] is the bandwidth adjustment.
NBayesFit

# ROC Plot using test set
#install.packages("pROC") #if not installed before
library(pROC)
roc_curve <- roc(test_data_k_n$output, predict(NBayesFit, newdata = test_data_k_n, type = "prob")$H)
plot(roc_curve, main = "ROC Curve", print.auc = TRUE)

#### 3.5.2 Trial with 6 selected predictor
train_data_k_n_reduced <- train_data_k_n[c('sex','cp','trtbps','slp','output')]
test_data_k_n_reduced <- test_data_k_n[c('sex','cp','trtbps','slp','output')]

NBayesFit_r <- train_data_k_n_reduced %>% train(output ~ .,
                                                  method = "nb",
                                                  data = .,
                                                  trControl = trainControl(
                                                    method = "cv", # used for configuring resampling method: in this case cross validation 
                                                    number = 10,
                                                    index = createFolds(train_data_k_n_reduced$output, k = 10),
                                                    classProbs = TRUE, 
                                                    verboseIter = FALSE,
                                                  ),
                                                  tuneLength = 10,
                                                  tuneGrid = expand.grid(
                                                    fL = 1:5,
                                                    usekernel = c(TRUE, FALSE),
                                                    adjust = 1:3
                                                  )
)

NBayesFit_r

# ROC Plot using test set
roc_curve <- roc(test_data_k_n_reduced$output, predict(NBayesFit_r, newdata = test_data_k_n_reduced, type = "prob")$H)
plot(roc_curve, main = "ROC Curve", print.auc = TRUE)

#### 3.5.3 Compare the models
# Compare the Accuracy and Kappa of the two models
resamps <- resamples(list(
  NB = NBayesFit,
  NBr= NBayesFit_r
))
resamps
bwplot(resamps, layout = c(3, 1))


## IV. Model Selection

### 4.1 checking accuracy and kappa
resamps <- resamples(list(
  DT  = rpartFit1,
  DT9 = rpartFit2,
  DT6 = rpartFit3,
  KNN = knn_model,
  KNN6 = knn_model6,
  KNN9 = knn_model9,
  SVM = svmFit,
  SVM6 = svmFit6,
  SVM9 = svmFit9,
  NB = NBayesFit,
  NBr = NBayesFit_r
))
resamps
bwplot(resamps, layout = c(3, 1))
##### Among the classifiers, SVM using 9 selected features work best
##### On the other hand, we have another finding in which 'Logistics Regression using 6 selected features also work well'
##### We put them both to Test Data to decide which is our FINAL SELECTED MODEL.


## V. Applying the Chosen Model(s) to the Test Data

### 5.1 Logistics Regression with 6 selected features
predicted_bw <- predict(model_bw, newdata = test_data_k)
predictedbw_labels <- ifelse(predicted_bw > 0.5, 1, 0)
accuracy_bw <- mean(predictedbw_labels == test_data_k$output)
print(paste("Accuracy:", accuracy_bw))

### 5.2 SVM using 9 selected features
predicted_svm9 <- predict(svmFit9, newdata = test_data_k_s)
accuracy_svm9 <- mean(predicted_svm9 == test_data_k_s$output)
print(paste("Accuracy:", accuracy_svm9))
##### Conclusion: Logistics Regression model is the best for the prediction of heart attacks


## 6. The Model and Final Validation

### 6.0 The model
##### The Logistic Regression Model "LR6"
coefficients <- coef(model_bw)
coefficients

##### What is the model?
##### The model is a two step mechanism:
##### 1. L = 3.6597 
#####        + (-1.9135) * sex 
#####        + 1.2943 * cp1 
#####        + 2.4267 * cp2 
#####        + 2.9993 * cp3 
#####        + (-0.0376) * trtbps 
#####        + (-0.1392) * slp1 
#####        + 1.9200 * slp2 
#####        + (-2.3307) * caa1 
#####        + (-3.4428) * caa2 
#####        + (-2.4312) * caa3 
#####        + 1.6098 * caa4 
#####        + 2.6753 * thall1 
#####        + 2.7076 * thall2 
#####        + 0.9987 * thall3
#####
##### 2. Y = 1 if L >= 0.5;
#####      = 0 if L < 0.5
#####
##### When the output is 1, we predict the new observation is an Heart-attack case; else, it is not.

### Additional validation by examine overall confusion matrix
confusion_matrix_bw_retry <- confusionMatrix(factor(predictedbw_labels),test_data_k$output, positive="1")
print(confusion_matrix_bw_retry)
#### Conclusion: Satisfactory accuracy, sensitivity, precision are spotted; The prediction model is reliable particularly in spotting "presence of heart attack" among patients.


