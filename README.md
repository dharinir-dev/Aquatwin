# AquaTwin — Water Consumption Prediction using Random Forest

## Overview

AquaTwin ML is the machine learning component of the AquaTwin project. It predicts water consumption in data-centre cooling systems using a **Random Forest Regression** model.

The model learns patterns from cooling-related parameters and predicts the expected water consumption, helping support water-efficiency analysis and smarter cooling management.

## Problem Statement

Data centres require continuous cooling to maintain safe operating temperatures. Cooling systems, especially evaporative cooling systems, can consume significant amounts of water.

AquaTwin ML uses machine learning to predict water consumption based on system and environmental parameters.

## Objectives

- Predict water consumption in data-centre cooling systems
- Analyze the relationship between cooling parameters and water usage
- Identify important features affecting water consumption
- Support data-driven water-efficiency decisions

## Machine Learning Model

### Random Forest Regression

The project uses **Random Forest Regressor** from Scikit-learn.

Random Forest combines multiple decision trees and averages their predictions to produce a more robust regression result.

### Why Random Forest?

- Handles non-linear relationships
- Works well with tabular data
- Handles multiple input features
- Robust to noise
- Requires minimal preprocessing
- Provides feature importance

## Input Features

The model can use parameters such as:

| Feature | Description |
|---|---|
| Temperature | System/environment temperature |
| Humidity | Environmental humidity |
| Cooling Load | Cooling demand of the system |
| Water Level | Available water level |
| Time | Time-related information |

> The exact features depend on the dataset used for training.

## Target Variable

**Water Consumption**

The model predicts the amount of water consumed by the cooling system.

## ML Workflow

```text
Dataset
   ↓
Data Preprocessing
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Random Forest Regression
   ↓
Model Training
   ↓
Prediction
   ↓
Model Evaluation