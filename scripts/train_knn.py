import os
import sys
import traceback
import argparse
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score
import joblib

def train():
    print("STARTING K-NN TRAINING...")
    
    # 1. Parse arguments from SageMaker
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--n-neighbors', type=int, default=5)
    args = parser.parse_args()

    # 2. Load Data
    # SageMaker K-NN expects the target variable in the FIRST column for CSV
    train_file = os.path.join(args.train, 'train.csv')
    print(f"Loading training data from: {train_file}")
    df = pd.read_csv(train_file)
    
    # Separate features (X) and target (y)
    # 'type' is our target. Everything else is a feature.
    y = df['type']
    X = df.drop(columns=['type', 'record_id', 'event_time']) # Drop metadata
    
    print(f"Training on {len(X)} samples with {len(X.columns)} features.")

    # 3. Train Model
    print("Training K-NN Classifier...")
    model = KNeighborsClassifier(n_neighbors=args.n_neighbors, metric='cosine')
    model.fit(X, y)
    
    # 4. Evaluate (Simple Accuracy/Recall proxy for the container log)
    y_pred = model.predict(X)
    # Using macro recall to handle class imbalance
    recall = recall_score(y, y_pred, average='macro', zero_division=0)
    print(f"Training Recall Score: {recall:.4f}")

    # 5. Save Model
    model_path = os.path.join(args.model_dir, 'model.joblib')
    print(f"Saving model to: {model_path}")
    joblib.dump(model, model_path)
    
    print("TRAINING COMPLETED SUCCESSFULLY.")

if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        error_msg = f"ERROR: {str(e)}\n\nTRACEBACK:\n{traceback.format_exc()}"
        print(error_msg)
        # Write to a known location so we can find it if it crashes
        with open('/opt/ml/output/failure', 'w') as f:
            f.write(error_msg)
        sys.exit(1)