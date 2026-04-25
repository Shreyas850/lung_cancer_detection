import pickle
from pathlib import Path
import xgboost as xgb
import optuna
import mlflow
from sklearn.metrics import accuracy_score, classification_report

from src.data_loader import load_data
from src.preprocess import clean_and_encode, prepare_splits

def optimize_hyperparameters(trial, x_train, y_train, x_test, y_test):
    """Optuna objective function to find the best XGBoost parameters."""
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        # Since you have a GTX 1650, you can uncomment the line below to train on GPU!
        # "device": "cuda", 
        "eval_metric": "logloss"
    }
    
    model = xgb.XGBClassifier(**param)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    return accuracy_score(y_test, preds)

def execute_pipeline():
    """Main execution thread with MLflow tracking."""
    data_path = Path("data/lung_cancer_dataset.csv")
    artifact_path = Path("models/lung_cancer.pkl")
    
    print("[SYSTEM] Loading and processing data...")
    raw_data = load_data(data_path)
    processed_data = clean_and_encode(raw_data)
    x_train, x_test, y_train, y_test = prepare_splits(processed_data)
    
    # Set up MLflow tracking
    mlflow.set_experiment("Lung_Cancer_Optimization")
    
    print("[SYSTEM] Starting Hyperparameter Optimization...")
    study = optuna.create_study(direction="maximize")
    # We will run 20 trials to find the best model configuration
    study.optimize(lambda trial: optimize_hyperparameters(trial, x_train, y_train, x_test, y_test), n_trials=20)
    
    best_params = study.best_params
    print(f"\n[SYSTEM] Best Parameters Found: {best_params}")
    
    print("\n[SYSTEM] Training Final Production Model...")
    with mlflow.start_run(run_name="Production_Candidate"):
        # Train the final model using the best parameters found by Optuna
        final_model = xgb.XGBClassifier(**best_params, eval_metric="logloss")
        final_model.fit(x_train, y_train)
        
        # Evaluate
        predictions = final_model.predict(x_test)
        final_accuracy = accuracy_score(y_test, predictions)
        print("\nClassification Report:\n", classification_report(y_test, predictions))
        
        # Log everything to MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", final_accuracy)
        
        # Save the physical artifact
        with open(artifact_path, "wb") as f:
            pickle.dump(final_model, f)
            
        print(f"[SYSTEM] Pipeline complete. Model saved to {artifact_path}")

if __name__ == "__main__":
    execute_pipeline()