# Gait-Based User Authentication

MATLAB machine learning system for user identification using walking patterns from wearable sensors.

## Overview

Identifies users based on unique gait patterns using accelerometer and gyroscope data. Extracts 74 features and trains 3 ML models (KNN, SVM, Neural Network) for authentication.

## Features

- 6-axis IMU sensor processing (accelerometer + gyroscope)
- 74 time/frequency domain features
- Multiple classifiers: KNN, SVM, FFNN
- Authentication metrics: FAR, FRR, EER
- Automated visualizations and analysis

## Quick Start

**Requirements:** MATLAB R2020b+, Statistics & Machine Learning Toolbox, Deep Learning Toolbox

**Run Pipeline:**

```matlab
% Execute all scripts in order
scripts = {'config', 'segment', 'preprocess_features', 'build_dataset', ...
           'train_models', 'evaluate', 'user_thresholds', ...
           'feature_analysis', 'generate_visualizations'};
for i = 1:length(scripts), run(scripts{i}); end
```

## Project Structure

```
├── config.m                  # Configuration
├── segment.m                 # Window segmentation
├── preprocess_features.m     # Feature extraction (74 features)
├── build_dataset.m           # Normalization & train/test split
├── train_models.m            # KNN, SVM, FFNN training
├── evaluate.m                # Metrics & confusion matrix
├── user_thresholds.m         # Per-user EER thresholds
├── feature_analysis.m        # PCA, ANOVA analysis
├── generate_visualizations.m # All plots
├── data/                     # 10 users × 2 sessions (FD/MD)
└── results/                  # Outputs (metrics, plots)
```

## Data Format

**Input:** CSV files with 6 columns (Accel XYZ, Gyro XYZ)  
**Naming:** `U<ID>NW_<Session>.csv` (e.g., `U1NW_FD.csv`)

- `FD` = First Day (training)
- `MD` = Middle Day (testing)

## Models

1. **KNN** - k-nearest neighbors with cross-validation
2. **SVM** - RBF kernel with ECOC multi-class
3. **FFNN** - Neural network [128-64-32] layers

## Metrics

- Classification: Accuracy, Precision, Recall, F1-Score
- Authentication: FAR, FRR, EER per user
- Visualizations: PCA, confusion matrix, feature importance

## Configuration

Edit `config.m`:

```matlab
params.window_len = 128;        % ~4s windows @ 32Hz
params.hop = 32;                % 75% overlap
params.normalize_method = 'minmax';
params.ffnn_layers = [128 64 32];
```

## Output

**Results folder contains:**

- `metrics.mat` - All performance metrics
- `confusion_ffnn.png` - Confusion matrix
- `pca_2d.png`, `pca_3d.png` - Feature space visualization
- `feature_importance.png` - Top features
- `accuracy_comparison.png` - Model performance



_Gait recognition using wearable sensor data | November 2025_
