import os
import time
import sagemaker
import sagemaker.estimator
import pandas as pd

def run_standalone_phase4():
    print("="*60)
    print(" PHASE 4: STANDALONE TRAIN & REGISTER (Bypassing Pipeline)")
    print("="*60)
    
    bucket = "company-mlops-netflix-production-592444418608"
    role_arn = "arn:aws:iam::592444418608:role/AmazonSageMaker-ExecutionRole-NetflixMLOps"
    sess = sagemaker.Session(default_bucket=bucket)
    region = sess.boto_region_name
    
    # 1. Prepare Data (We know this works from your earlier runs)
    print("\n STEP 1: Preparing Training Data...")
    df = pd.read_csv(f"s3://{bucket}/processed/processed_data.csv")
    tfidf_cols = [c for c in df.columns if c.startswith('tfidf_feat_')]
    feature_cols = tfidf_cols + ['release_year']
    
    df_model = df[['type'] + feature_cols].copy()
    df_model['type'] = df_model['type'].map({'Movie': 1, 'TV Show': 0}).fillna(0).astype(int)
    df_model['release_year'] = pd.to_numeric(df_model['release_year'], errors='coerce').fillna(0).astype(int)
    df_model = df_model.fillna(0)
    
    os.makedirs("data", exist_ok=True)
    train_path = "data/standalone_train.csv"
    df_model.to_csv(train_path, index=False, header=False)
    
    s3_train_uri = sess.upload_data(path=train_path, bucket=bucket, key_prefix="standalone-train")
    print(f"✅ Data uploaded to: {s3_train_uri}")
    
    # 2. Train Model (Standalone Estimator - we know this works)
    print("\n STEP 2: Training XGBoost Model (This takes ~5-8 mins)...")
    xgb_image_uri = sagemaker.image_uris.retrieve("xgboost", region, version="1.7-1")
    
    estimator = sagemaker.estimator.Estimator(
        image_uri=xgb_image_uri,
        role=role_arn,
        instance_count=1,
        instance_type="ml.m5.large",
        output_path=f"s3://{bucket}/artifacts",
        sagemaker_session=sess,
        hyperparameters={
            "max_depth": "5",
            "eta": "0.2",
            "gamma": "4",
            "min_child_weight": "6",
            "objective": "binary:logistic",
            "num_round": "50",
            "eval_metric": "accuracy"
        }
    )
    
    estimator.fit({"train": sagemaker.inputs.TrainingInput(s3_train_uri, content_type="text/csv")})
    print("✅ Training Completed!")
    
    # 3. Register Model
    print("\n STEP 3: Registering Model in SageMaker Model Registry...")
    model_package = estimator.register(
        model_package_group_name="NetflixMLOpsModelGroup",
        approval_status="PendingManualApproval",
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"]
    )
    
    print(f"✅ Model Registered Successfully!")
    print(f"Model Package ARN: {model_package.model_package_arn}")
    print("="*60)
    print("🎉 PHASE 4 TRAIN & REGISTER COMPLETE!")
    print("Next: Approve the model and run deploy_endpoint.py")
    print("="*60)

if __name__ == '__main__':
    try:
        run_standalone_phase4()
    except Exception as e:
        print(f"\n❌ Standalone Phase 4 failed: {e}")