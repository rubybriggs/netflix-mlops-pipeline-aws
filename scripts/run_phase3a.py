import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
import pandas as pd
import os

def run_training():
    bucket = "company-mlops-netflix-production-592444418608"
    role_arn = "arn:aws:iam::592444418608:role/AmazonSageMaker-ExecutionRole-NetflixMLOps"
    sess = sagemaker.Session(default_bucket=bucket)
    region = sess.boto_region_name

    print("STEP 1: Preparing Training Data (Numeric Only)...")
    
    # 1. Load the processed data from Phase 2
    df = pd.read_csv(f"s3://{bucket}/processed/processed_data.csv")
    
    # 2. Isolate ONLY numeric features and the target
    # Identify the 100 TF-IDF columns we generated
    tfidf_cols = [col for col in df.columns if col.startswith('tfidf_feat_')]
    
    # We will use 'type' as the target, and tfidf + release_year as features
    feature_cols = tfidf_cols + ['release_year']
    
    # Create a clean dataframe with target FIRST (required by SageMaker K-NN)
    df_train = df[['type'] + feature_cols].copy()
    
    # 3. Map target to numeric (Movie = 1, TV Show = 0)
    df_train['type'] = df_train['type'].map({'Movie': 1, 'TV Show': 0}).fillna(0).astype(int)
    
    # 4. Ensure release_year is numeric, fill missing with 0
    df_train['release_year'] = pd.to_numeric(df_train['release_year'], errors='coerce').fillna(0).astype(int)
    
    # 5. Fill any remaining NaNs in TF-IDF features with 0 (safety net)
    df_train = df_train.fillna(0)
    
    # Save locally
    train_data_path = "data/train/train.csv"
    os.makedirs("data/train", exist_ok=True)
    df_train.to_csv(train_data_path, index=False, header=False) # K-NN often prefers no header, but header is usually fine. Let's keep no header to be safe.
    print(f"Saved {len(df_train)} rows with {len(feature_cols)} numeric features to {train_data_path}")

    # Upload to S3
    s3_train_prefix = "s3://{}/train".format(bucket)
    sess.upload_data(path="data/train/train.csv", bucket=bucket, key_prefix="train")
    print(f"Uploaded training data to {s3_train_prefix}")

    print("\nSTEP 2: Configuring SageMaker K-NN Estimator...")
    # Get the official AWS K-NN ECR image URI
    knn_image_uri = sagemaker.image_uris.retrieve("knn", region, version="1")
    
    estimator = Estimator(
        image_uri=knn_image_uri,
        role=role_arn,
        instance_count=1,
        instance_type="ml.m5.large",
        output_path="s3://{}/artifacts".format(bucket),
        sagemaker_session=sess,
        hyperparameters={
            "k": 10, 
            "sample_size": str(len(df_train)), 
            "predictor_type": "classifier",
            "feature_dim": str(len(feature_cols)) # Explicitly tell K-NN how many features to expect
        }
    )

    print("\nSTEP 3: Starting Training Job (This takes ~5-10 mins)...")
    # Note: content_type="text/csv;label_size=1" tells SageMaker the first column is the label
    estimator.fit({"train": TrainingInput(s3_train_prefix, content_type="text/csv;label_size=1")})
    
    print("\n" + "="*50)
    print("PHASE 3A TRAINING COMPLETE!")
    print("="*50)

if __name__ == '__main__':
    run_training()