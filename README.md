# Tpsm-analysis

# Data-Driven Decision Making and Organizational Confidence: A Statistical Modelling Study

## Overview

This project investigates the relationship between data-driven decision-making capabilities and organizational confidence in business strategy. Using a synthetic dataset of over 1,000 companies and 18 business analytics variables, the study applies descriptive, inferential, and predictive analytics techniques to determine whether higher levels of data maturity lead to greater confidence in strategic decision-making.

The dataset was designed using concepts derived from the PwC Global Data & Analytics Survey and the McKinsey Analytics Maturity Model framework.

## Research Question

**Does data-driven decision making improve organizational confidence in business strategies?**

The findings provide strong statistical evidence that organizations with greater data maturity, analytics adoption, and data-driven culture exhibit significantly higher confidence in their strategic decisions.

## Technologies Used

* R Programming
* ggplot2
* dplyr
* caret
* randomForest
* gbm
* glmnet
* corrplot

## Analytics Performed

### Descriptive Analytics

* Summary statistics
* Frequency distributions
* Group comparisons
* Data visualizations
* Organizational confidence profiling

### Inferential Analytics

* Pearson Correlation Analysis
* Simple Linear Regression
* Multiple Linear Regression
* One-Way ANOVA
* Independent Sample t-Tests

### Predictive Analytics

Five machine learning models were developed and compared:

1. Multiple Linear Regression
2. Ridge Regression
3. Decision Tree
4. Random Forest
5. Gradient Boosting Machine (GBM)

Models were evaluated using:

* R² (Coefficient of Determination)
* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* 5-Fold Cross-Validation

## Key Findings

### Correlation Analysis

* Pearson Correlation: **r = 0.954**
* Strong positive relationship between data maturity and organizational confidence.

### Simple Linear Regression

* **R² = 0.91**
* Data maturity alone explains approximately 91% of the variation in organizational confidence.
* Each one-level increase in data maturity predicts an average increase of approximately 17.8 percentage points in confidence.

### ANOVA Results

* **F = 1059**
* **p < 0.001**
* Organizations at the highest maturity level demonstrated dramatically higher confidence scores than organizations at the lowest maturity level.

### Independent t-Tests

* Real-time data access associated with approximately **38 percentage points** higher confidence.
* Dedicated data teams associated with approximately **20 percentage points** higher confidence.
* Both differences were statistically significant (**p < 0.001**).

### Predictive Modelling Results

**Best Performing Model: Multiple Linear Regression**

Performance Metrics:

* Test R² = 0.91
* MAE = 5.4 percentage points
* 5-Fold Cross-Validation R² = 0.917 ± 0.008

The model demonstrated strong predictive accuracy while maintaining interpretability and showing no evidence of overfitting.

## Conclusion

The results suggest that investments in data infrastructure, analytics tools, real-time information access, data literacy, and executive training significantly improve organizational confidence in strategic decision-making. Data maturity is a critical driver of business confidence and competitive advantage in modern organizations.

## Project Structure

* Data Preparation
* Descriptive Analytics
* Inferential Analytics
* Predictive Modelling
* Model Evaluation
* Cross-Validation
* Scenario-Based Forecasting

## Author

**Avishka Kanakasekara**
BSc (Hons) in Information Technology Specializing in Data Science
Sri Lanka Institute of Information Technology (SLIIT)
