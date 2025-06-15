# --- FINAL SCRIPT FOR KAGGLE SUBMISSION: TITANIC ---
# Algorithm: Random Forest (implementation with the 'ranger' package)

# 1. LOAD LIBRARIES
# install.packages("ranger") # Make sure this is installed
# install.packages("tidyverse") # Make sure this is installed
library(ranger)
library(tidyverse)

# 2. LOAD DATA
train_data <- read.csv("train.csv")
test_data <- read.csv("test.csv")

# 3. DATA PROCESSING AND CLEANING
# Combine datasets for consistent processing
test_data$Survived <- NA
full_data <- rbind(train_data, test_data)

# Robust imputation of missing or non-finite numeric values
median_age <- median(full_data$Age, na.rm = TRUE)
full_data$Age[!is.finite(full_data$Age)] <- median_age

median_fare <- median(full_data$Fare, na.rm = TRUE)
full_data$Fare[!is.finite(full_data$Fare)] <- median_fare

# Imputation of missing values in categorical features
mode_embarked <- names(sort(table(full_data$Embarked), decreasing = TRUE))[1]
full_data$Embarked[full_data$Embarked == "" | is.na(full_data$Embarked)] <- mode_embarked

# 4. FEATURE ENGINEERING
# Create "Title" variable from Name
full_data$Title <- gsub('(.*, )|(\\..*)', '', full_data$Name)
rare_titles <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')
full_data$Title[full_data$Title %in% rare_titles] <- "Rare"
full_data$Title[full_data$Title %in% c('Mlle', 'Ms')] <- 'Miss'
full_data$Title[full_data$Title == 'Mme'] <- 'Mrs'

# Create "FamilySize" variable
full_data$FamilySize <- full_data$SibSp + full_data$Parch + 1

# Convert variables to factor type
full_data$Sex <- as.factor(full_data$Sex)
full_data$Embarked <- as.factor(full_data$Embarked)
full_data$Title <- as.factor(full_data$Title)
full_data$Pclass <- as.factor(full_data$Pclass)

# 5. FINAL DATA SPLIT
# Split back into training and test sets
train_final <- full_data[!is.na(full_data$Survived), ]
train_final$Survived <- as.factor(train_final$Survived) # Convert Survived to a factor

test_final <- full_data[is.na(full_data$Survived), ]

# 6. TRAIN THE 'RANGER' MODEL
# Define predictor variables to use
features <- c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title", "FamilySize")

# Set a seed for reproducibility
set.seed(123)

# Train the model
model_ranger <- ranger(
  dependent.variable.name = "Survived",
  data = train_final[, c("Survived", features)],
  num.trees = 500,
  mtry = 3,
  importance = "permutation"
)

# 7. PREDICTION AND SUBMISSION FILE CREATION
# Make predictions on the test set
prediction_object <- predict(model_ranger, data = test_final)

# Extract the predictions
final_predictions <- prediction_object$predictions

# Create the dataframe for Kaggle submission
submission <- data.frame(PassengerId = test_final$PassengerId, Survived = final_predictions)

# Save the file in .csv format
write.csv(submission, "submission_titanic_ranger.csv", row.names = FALSE)

# Confirmation message
print("'submission_titanic_ranger.csv' file created successfully.")