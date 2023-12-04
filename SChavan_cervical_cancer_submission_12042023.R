############################################################
# Cervical Cancer dataset 
###############################################
# Exploratory data analysis
# Load necessary libraries
# Load the 'caret' library if not already installed
library(caret)
if (!require(caret)) {
  install.packages("caret", repos = "http://cran.us.r-project.org")
  library(caret)
}

# Load the 'dplyr' library if not already installed
library(dplyr)
if (!require(dplyr)) {
  install.packages("dplyr", repos = "http://cran.us.r-project.org")
  library(dplyr)
}

# Load your dataset (ensure the file path is correct)
#link to dataset https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors
cervical_cancer <- read.csv("risk_factors_cervical_cancer.csv")

# Convert "?" to NA
cervical_cancer[cervical_cancer == "?"] <- NA

# Ensure 'Biopsy' is a factor variable with two levels
cervical_cancer$Biopsy <- as.factor(cervical_cancer$Biopsy)

# Drop columns with special characters
cervical_cancer <- cervical_cancer %>% select(-c(STDs..Time.since.first.diagnosis, STDs..Time.since.last.diagnosis))

# Remove rows with missing values
cervical_cancer <- na.omit(cervical_cancer)

summary(cervical_cancer)
class(cervical_cancer)

# Set option to display all columns
options(repr.matrix.max.cols=Inf, repr.matrix.max.rows=5, repr.max.print=Inf)

# Display the head of the cervical_cancer dataframe
head(cervical_cancer)

# Get the column names of the cervical_cancer dataframe
column_names <- names(cervical_cancer)

# Display the list of column names
print(column_names)
####################################################################
# Exploring the dataset
# Boxplot of diagnosis distribution by age using ggplot2
# Load the 'ggplot2' library if not already installed
if (!require(ggplot2)) {
  install.packages("ggplot2", repos = "http://cran.us.r-project.org")
  library(ggplot2)
}
library(ggplot2)

ggplot(cervical_cancer, aes(x = factor(Biopsy), y = Age, fill = factor(Biopsy))) +
  geom_boxplot() +
  scale_fill_manual(values = c("lightblue", "lightgreen")) +
  labs(title = "Boxplot of Diagnosis Distribution by Age",
       x = "Biopsy", y = "Age")
###################################################################################
# Countplots for risk factors
# Load necessary libraries
# Load the 'ggplot2' library if not already installed
if (!require(ggplot2)) {
  install.packages("ggplot2", repos = "http://cran.us.r-project.org")
  library(ggplot2)
}
library(ggplot2)

# Load the 'gridExtra' library if not already installed
if (!require(gridExtra)) {
  install.packages("gridExtra", repos = "http://cran.us.r-project.org")
  library(gridExtra)
}

library(gridExtra)

# Create count plots for risk factors with different colors
options(repr.plot.width = 15, repr.plot.height = 4)  # Set plot size

# Countplot for 'STDs'
p1 <- ggplot(cervical_cancer, aes(x = STDs)) +
  geom_bar(fill = c('salmon', 'lightblue'), color = 'black') +
  labs(title = 'Count Plot of STDs')

# Countplot for 'Smokes'
p2 <- ggplot(cervical_cancer, aes(x = Smokes)) +
  geom_bar(fill = c('salmon', 'lightblue'), color = 'black') +
  labs(title = 'Count Plot of Smokes')

# Countplot for 'Hormonal Contraceptives'
p3 <- ggplot(cervical_cancer, aes(x = `Hormonal.Contraceptives`)) +
  geom_bar(fill = c('salmon', 'lightblue'), color = 'black') +
  labs(title = 'Count Plot of Hormonal Contraceptives')

grid.arrange(p1, p2, p3, ncol = 3)
####################################################################################
# Data splitting and Preprocessing
##################################################################
# Create the matrix of features and the dependent variable vector
X <- cervical_cancer[, !colnames(cervical_cancer) %in% "Biopsy"]
y <- cervical_cancer[, "Biopsy"]

# Data set splitting
set.seed(1)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
y_train <- y[train_index]
X_test <- X[-train_index, ]
y_test <- y[-train_index]

##################################################################################
# Model testing
####################################################
# KNN model for training dataset
#############################################
# Load necessary libraries
# Load the 'caret' library if not already installed
if (!require(caret)) {
  install.packages("caret", repos = "http://cran.us.r-project.org")
  library(caret)
}
library(caret)

# Create and train a KNN model
knn_model <- train(
  x = X_train,
  y = y_train,
  method = "knn",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = expand.grid(k = 5)
)

# Print the model
print(knn_model)
##################################################################
# KNN model for test dataset
####################################################

# Predict on the test set
knn_predictions <- predict(knn_model, newdata = X_test)

# Confusion matrix
confusion_matrix <- confusionMatrix(knn_predictions, y_test)
print(confusion_matrix)

# Model performance metrics
accuracy <- confusion_matrix$overall["Accuracy"]
precision <- confusion_matrix$byClass["Precision"]
recall <- confusion_matrix$byClass["Recall"]
f1_score <- confusion_matrix$byClass["F1"]

# Print performance metrics
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
########################################################################
### Support Vector Machines (SVM) model
####################################################
# Load the 'caret' library if not already installed
if (!require(caret)) {
  install.packages("caret", repos = "http://cran.us.r-project.org")
  library(caret)
}
library(caret)
# Load the 'e1071' library if not already installed
if (!require(e1071)) {
  install.packages("e1071", repos = "http://cran.us.r-project.org")
  library(e1071)
}
library(e1071)  # for SVM
# Create and train an SVM model
svm_model <- svm(
  x = X_train,
  y = y_train,
  kernel = "linear",  # you can try different kernels (linear, polynomial, etc.)
  cost = 1,          # cost parameter (adjust as needed)
  scale = FALSE      # you can adjust other parameters based on your requirements
)

# Print the model
print(svm_model)

# Predict on the training set 
train_predictions <- predict(svm_model, newdata = X_train)

# Print training accuracy 
train_accuracy <- confusionMatrix(train_predictions, y_train)$overall["Accuracy"]
cat("Training Accuracy:", train_accuracy, "\n")

# Predict on the test set
test_predictions <- predict(svm_model, newdata = X_test)

# Confusion matrix
confusion_matrix_svm <- confusionMatrix(test_predictions, y_test)
print(confusion_matrix_svm)

# Model performance metrics
accuracy_svm <- confusion_matrix_svm$overall["Accuracy"]

# Print test accuracy
cat("Test Accuracy (SVM):", accuracy_svm, "\n")

####################################################
# Random Forest model for training dataset
####################################################
# Load necessary libraries
# Load the 'caret' library if not already installed
if (!require(caret)) {
  install.packages("caret", repos = "http://cran.us.r-project.org")
  library(caret)
}
library(caret)
# Create and train a random forest model
rf_model <- train(
  x = X_train,
  y = y_train,
  method = "rf",  # Random forest method
  trControl = trainControl(method = "cv", number = 10)  # Cross-validation
)

# Print the model
print(rf_model)

####################################################
# Random Forest model evaluation on the test set
####################################################

# Predict on the test set
rf_predictions <- predict(rf_model, newdata = X_test)

# Confusion matrix
confusion_matrix_rf <- confusionMatrix(rf_predictions, y_test)
print(confusion_matrix_rf)

# Model performance metrics
accuracy_rf <- confusion_matrix_rf$overall["Accuracy"]

# Print test accuracy
cat("Test Accuracy (Random Forest):", accuracy_rf, "\n")

################################################################
### ROC curve and Area Under the Curve (AUC)
#############################################################
# Load necessary libraries
# Load the 'caret' library if not already installed
if (!require(caret)) {
  install.packages("caret", repos = "http://cran.us.r-project.org")
  library(caret)
}
library(caret)
# Load the 'pROC' library if not already installed
if (!require(pROC)) {
  install.packages("pROC", repos = "http://cran.us.r-project.org")
  library(pROC)
}
library(pROC)
####################################################
# ROC KNN model for test dataset
####################################################
# Set the layout to a single plot
par(mfrow = c(1, 1))

# Create a new single plot (replace this with your own plot code)
new_plot <- ggplot(...) + ...

# Print the new plot
print(new_plot)

# Predict on the test set
knn_predictions <- predict(knn_model, newdata = X_test)

# Calculate predicted probabilities
knn_probabilities <- as.numeric(predict(knn_model, newdata = X_test, type = "prob")[, "1"])

# Create ROC curve for KNN
roc_knn <- roc(y_test, knn_probabilities)

####################################################
# ROC SVM model for test dataset
####################################################

# Predict on the test set
svm_predictions <- predict(svm_model, newdata = X_test)

# Calculate predicted probabilities
svm_probabilities <- as.numeric(predict(svm_model, newdata = X_test, type = "response"))

# Create ROC curve for SVM
roc_svm <- roc(y_test, svm_probabilities)

####################################################
# ROC Random Forest model for test dataset
####################################################

# Predict on the test set
rf_predictions <- predict(rf_model, newdata = X_test)

# Calculate predicted probabilities
rf_probabilities <- predict(rf_model, newdata = X_test, type = "prob")[, "1"]

# Create ROC curve for Random Forest
roc_rf <- roc(y_test, rf_probabilities)

####################################################
# Plotting ROC curves for all three models
####################################################

# Plot ROC curves
plot(roc_knn, col = "blue", main = "ROC Curves for Model Comparison")
lines(roc_svm, col = "red")
lines(roc_rf, col = "green")

#####################################
## Area Under Curve (AUC)
# Load the 'pROC' library if not already installed
if (!require(pROC)) {
  install.packages("pROC", repos = "http://cran.us.r-project.org")
  library(pROC)
}
library(pROC)

####################################################
# AUC KNN model for test dataset
####################################################

# Predict on the test set
knn_predictions <- predict(knn_model, newdata = X_test)

# Calculate predicted probabilities
knn_probabilities <- as.numeric(predict(knn_model, newdata = X_test, type = "prob")[, "1"])

# Create ROC curve for KNN
roc_knn <- roc(y_test, knn_probabilities)

####################################################
# AUC SVM model for test dataset
####################################################

# Predict on the test set
svm_predictions <- predict(svm_model, newdata = X_test)

# Calculate predicted probabilities
svm_probabilities <- as.numeric(predict(svm_model, newdata = X_test, type = "response"))

# Create ROC curve for SVM
roc_svm <- roc(y_test, svm_probabilities)

####################################################
# AUC Random Forest model for test dataset
####################################################

# Predict on the test set
rf_predictions <- predict(rf_model, newdata = X_test)

# Calculate predicted probabilities
rf_probabilities <- predict(rf_model, newdata = X_test, type = "prob")[, "1"]

# Create ROC curve for Random Forest
roc_rf <- roc(y_test, rf_probabilities)

####################################################
# Compute AUC for each model
####################################################

auc_knn <- auc(roc_knn)
auc_svm <- auc(roc_svm)
auc_rf <- auc(roc_rf)

# Print AUC values
cat("AUC for KNN:", auc_knn, "\n")
cat("AUC for SVM:", auc_svm, "\n")
cat("AUC for Random Forest:", auc_rf, "\n")
############################################################
  
  