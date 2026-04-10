# Model Training and Evaluation Workflow

This directory contains the scripts and artifacts for training, validating, and evaluating multiple machine learning classifiers on the [preprocessed dataset](../datasets/preprocessed_splits/). 

## Directory Structure

```bash
(models/)
├── AutoGluon/
│   ├── 1-train.py
│   ├── 2-test.py
│   └── model_final_AutoGluon/      # AutoGluon saves a folder, not a .pkl file
├── CatBoost/
│   ├── 1-train_cv.py
│   ├── 2-train_final.py
│   ├── 3-test.py
│   ├── classification_report_test.txt
│   ├── confusion_matrix_test.csv
│   └── model_final_CatBoost.pkl
├── LogisticRegression/
│   └── ... (same as above)
├── RandomForest/
│   └── ... (same as above)
├── SVC/
│   └── ... (same as above)
└── XGBoost/
    └── ... (same as above)    
```

Each subfolder contains all scripts and outputs for a single model family.

----

## Workflow Overview

For each model family (e.g. XGBoost, RandomForest, etc.), you will find:

1. 1-train_cv.py:  
    Runs cross-validation and hyperparameter search within the training set, logs results to MLflow.

2. 2-train_final.py:  
    Retrains the model using the best parameters from CV on all of the training data. This produces the final model for testing.

3. 3-test.py:  
    Loads the final model, runs predictions on the independent test set, computes metrics and confusion matrix, logs to MLflow.

4. model_final_<Model>.pkl:  
    The final trained model (except AutoGluon, which stores a folder).

5. classification_report_test.txt / confusion_matrix_test.csv:  
    Artifacts with performance results on test data.

----

## How to Train and Evaluate a Model

Step 1: Cross-validation and Parameter Search


    cd models/<ModelName>
    python3 1-train_cv.py


- This logs all CV runs, folds, hyperparameters, and metrics to MLflow (experiment multiclass_model_selection).

Step 2: Train the Final Model

    python3 2-train_final.py


- This script automatically selects the best configuration from CV, retrains the model, and logs the final model in MLflow.

Step 3: Test and Evaluate
    
    python3 3-test.py

- This script generates predictions on the test set, computes performance metrics, saves artifacts (classification_report_test.txt, confusion_matrix_test.csv), and logs all results in MLflow (experiment final_evaluation).

----

## Visualizing Results with MLflow

1. Start the MLflow UI:
       mlflow ui

   By default: go to http://localhost:5000

2. View Experiments and Compare Models:
    - Select multiclass_model_selection for all training and CV runs.
    - Select final_evaluation to see final test results for each model.
    - You can sort, filter, and compare by model type, metrics, run names, etc.
    - All artifacts (models, confusion matrices, classification reports) are accessible for download on each run's detail page.

----

## Adding Models or Extending

- Add a new folder under models/ (e.g. LightGBM/)
- Copy/adapt the three script templates.
- Make sure to use the same MLflow experiment names and tags for comparison!

----

## Notes

- Data: Train/test CSVs must be preprocessed and available under datasets/preprocessed_splits/
- Requirements: See requirements.txt for dependencies—install them in your environment.
- AutoGluon: Stores the model as a folder, not a .pkl. Load using TabularPredictor.load(...)

