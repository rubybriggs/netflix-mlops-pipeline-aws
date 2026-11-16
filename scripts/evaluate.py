import os
import sys
import json
import traceback
import tarfile
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

def evaluate():
    print("STARTING MODEL EVALUATION...")
    
    model_path = "/opt/ml/processing/model/model.tar.gz"
    test_path = "/opt/ml/processing/test/train.csv"
    output_dir = "/opt/ml/processing/evaluation"
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. Extract Model (Built-in XGBoost saves as .tar.gz)
        print(f"Extracting model from {model_path}")
        extract_dir = "/opt/ml/processing/model/extracted"
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(model_path, 'r:gz') as tar:
            tar.extractall(path=extract_dir)
        
        model_file = os.path.join(extract_dir, "xgboost-model")
        
        # 2. Load Test Data
        print(f"Loading test data from {test_path}")
        df_test = pd.read_csv(test_path, header=None)
        y_test = df_test.iloc[:, 0]
        X_test = df_test.iloc[:, 1:]
        
        # 3. Load Model & Predict
        print("Loading trained model...")
        model = xgb.Booster()
        model.load_model(model_file)
        
        print("Running predictions...")
        dtest = xgb.DMatrix(X_test)
        y_pred_probs = model.predict(dtest)
        y_pred = [round(value) for value in y_pred_probs]
        
        acc = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {acc:.4f}")
        
        # 4. Save Evaluation Report
        report_dict = {
            "classification_metrics": {
                "accuracy": {"value": acc, "standard_deviation": "NaN"}
            }
        }
        
        output_file = os.path.join(output_dir, "evaluation.json")
        with open(output_file, "w") as f:
            json.dump(report_dict, f, indent=2)
            
        print(f"Evaluation report saved to {output_file}")
        print("EVALUATION COMPLETED SUCCESSFULLY.")

    except Exception as e:
        error_msg = f"ERROR: {str(e)}\n\nTRACEBACK:\n{traceback.format_exc()}"
        print(error_msg)
        with open('/opt/ml/output/failure', 'w') as f:
            f.write(error_msg)
        sys.exit(1)

if __name__ == '__main__':
    evaluate()