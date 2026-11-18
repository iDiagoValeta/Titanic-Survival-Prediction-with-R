# Titanic Survival Prediction with R

**A Random Forest model in R that predicts which Titanic passengers survived, built as an entry for the Kaggle "Titanic: Machine Learning from Disaster" competition.**

> **Context.** Built in June 2025 as a personal project, when I was a Computer Engineering student at UPV and a guide at the UPV Computer Museum, in my first months of public repositories (phase of machine learning coursework and practice). It is one of my first end-to-end machine learning exercises, kept as it was written; the code is not maintained.

## What it does

- Cleans the Kaggle Titanic `train.csv` and `test.csv` (median imputation for `Age` and `Fare`, mode for `Embarked`).
- Engineers two features: `Title`, extracted from the passenger name with rare titles grouped, and `FamilySize` (`SibSp` + `Parch` + 1).
- Trains a Random Forest with the `ranger` package (500 trees, `mtry = 3`, permutation importance).
- Writes `submission_titanic_ranger.csv`, ready to upload to Kaggle.
- Ships a PDF report (`titanic.pdf`) with the exploratory analysis, the modeling decisions and the variable importance.

## How it works

`titanic.R` merges train and test so both get identical preprocessing, imputes missing values, builds `Title` and `FamilySize`, converts `Pclass`, `Sex`, `Embarked` and `Title` to factors, and splits the data again. It fits a `ranger` classifier on nine predictors (`Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`, `Title`, `FamilySize`) with seed 123, predicts the test set and saves the submission file.

The report explains that the first model used `randomForest` (with `caret` loaded), that `predict()` on the test set showed persistent anomalies despite data validation, and that the final script therefore switched to `ranger`. The report's exploratory section looks at survival by sex, ticket class and age.

## Results

Both figures come from `titanic.pdf`, not from re-running the code:

- Out-of-bag error of the first `randomForest` model: 16.39 % (about 83.61 % accuracy), with 90.1 % of non-survivors and 73.1 % of survivors classified correctly.
- Kaggle score of the final `ranger` submission: 77.03 % accuracy.
- Most important variables, in order: `Title`, `Fare`, `Sex`, `Pclass`.

## Run it

Not re-run for this documentation. The script expects R with the `ranger` and `tidyverse` packages and `train.csv` and `test.csv` in the working directory (both are in the repo; `titanic.zip` holds the original Kaggle files).

```r
install.packages(c("ranger", "tidyverse"))
setwd("path/to/this/repo")
source("titanic.R")   # writes submission_titanic_ranger.csv
```

## Repository layout

```
titanic.R                      full pipeline: cleaning, features, ranger model, submission
titanic.pdf                    report: EDA, decisions, OOB results, importance
train.csv, test.csv            Kaggle data
titanic.zip                    original Kaggle download (train, test, gender_submission)
submission_titanic_ranger.csv  predictions in Kaggle format
```

## Stack

R, tidyverse, ranger (the report also used randomForest and caret), Kaggle.
