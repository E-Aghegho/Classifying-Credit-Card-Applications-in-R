# Credit Card Fraud Detection Using KNN and SVM.

## Overview
This project explores the classification of credit card transactions using k-Nearest Neighbors (KNN) and Support Vector Machines (SVM). The dataset used is `credit_card_data-headers.txt`, and cross-validation techniques were applied to optimize model performance.

## Data Preparation
- The dataset was split into training (80%) and test (20%) sets.
- For KNN, the training set was further divided into 10 folds for cross-validation.
- For SVM, an additional validation set (15%) was created.

## K-Nearest Neighbors (KNN) Model
### Cross-Validation Strategy
1. The dataset was divided into 10 folds.
2. The model was trained on 9 folds and validated on the remaining fold.
3. This process was repeated 10 times, with each fold serving as validation once.
4. Accuracy was calculated for different values of `k`.
5. The best `k` value was chosen based on maximum accuracy.
6. The final model was trained using the entire dataset with the best `k`.

### Implementation
```r
# Load dataset
data <- read.table("credit_card_data-headers.txt", stringsAsFactors = FALSE, header = TRUE)
set.seed(2)

# Split data into training (80%) and test (20%)
split_data <- sample(c("Train", "Test"), nrow(data), replace = TRUE, prob = c(0.8, 0.2))
train_set <- data[split_data == "Train", ]
test_set <- data[split_data == "Test", ]

# Separate features and response variable
response <- train_set[, 11]
train_data <- train_set[, -11]

# Create 10 folds
folds <- createFolds(response, k = 10, list = TRUE, returnTrain = FALSE)

# Find optimal k
k_values <- seq(1, 100, by = 1)
k_accuracies <- numeric(length(k_values))

for (j in seq_along(k_values)) {
  fold_accuracies <- numeric(length(folds))
  for (i in seq_along(folds)) {
    train_indices <- unlist(folds[-i])
    validate_indices <- folds[[i]]
    train <- train_data[train_indices, ]
    train_response <- response[train_indices]
    validate <- train_data[validate_indices, ]
    validate_response <- response[validate_indices]
    predicted <- knn(train = train, test = validate, cl = train_response, k = k_values[j])
    fold_accuracies[i] <- sum(predicted == validate_response) / length(validate_response)
  }
  k_accuracies[j] <- mean(fold_accuracies)
}

# Best k value
best_k <- k_values[which.max(k_accuracies)]

# Train and test final model
predicted_labels <- knn(train = train_data, test = test_set[, -11], cl = response, k = best_k)
accuracy <- sum(predicted_labels == test_set[, 11]) / length(test_set[, 11])
print(accuracy)
```
### Results
- Best k value found: `best_k`
- Final test accuracy: **68%**

## Support Vector Machine (SVM) Model
### Implementation
```r
library(kernlab)

# Load dataset
data <- read.table("credit_card_data-headers.txt", stringsAsFactors = FALSE, header = TRUE)
set.seed(2)

# Split data into training (70%), validation (15%), and test (15%)
split_data <- sample(c("Train", "Validate", "Test"), nrow(data), replace = TRUE, prob = c(0.7, 0.15, 0.15))
train_set <- data[split_data == "Train", ]
validate_set <- data[split_data == "Validate", ]
test_set <- data[split_data == "Test", ]

# Define range of C values
C_values <- 10^seq(-3, 7, by = 0.5)
accuracies <- numeric(length(C_values))

# Train SVM with different C values
for (i in seq_along(C_values)) {
  C <- C_values[i]
  model_SVM <- ksvm(as.matrix(train_set[,1:10]), as.factor((train_set[,11])), type = "C-svc", kernel = "vanilladot", C = C, scaled = TRUE)
  pred <- predict(model_SVM, validate_set[, 1:10])
  accuracies[i] <- sum(pred == validate_set[, 11]) / nrow(validate_set)
}

# Best C value
best_C <- C_values[which.max(accuracies)]
best_model_SVM <- ksvm(as.matrix(train_set[,1:10]), as.factor((train_set[,11])), type = "C-svc", kernel = "vanilladot", C = best_C, scaled = TRUE)

# Test final model
bm_pred <- predict(best_model_SVM, test_set[, 1:10])
accuracy <- sum(bm_pred == test_set[, 11]) / nrow(test_set)
print(accuracy)
```
### Results
- Best C value found: `best_C`
- Final test accuracy: **88%**

### SVM Model Summary
```
Support Vector Machine object of class "ksvm"

SV type: C-svc  (classification)
parameter : cost C = 0.01

Linear (vanilla) kernel function.

Number of Support Vectors : 232

Objective Function Value : -1.8338
Training error : 0.147903
```
### Predictions
```
Predicted Labels:
0 0 1 0 0 1 0 1 0 1 0 1 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 1 0 1 1 0 0 1 0 1 1 1 0 1 1 0 1 1 1 1 1 1 1 1
...
Accuracy: 87.88%
```

## Conclusion
- KNN achieved **68% accuracy**, indicating possible errors in methodology or hyperparameter tuning.
- SVM outperformed KNN with an **88% accuracy**, suggesting it is a more reliable model for this dataset.
- Further improvements can be made by experimenting with different kernel functions and feature engineering.
