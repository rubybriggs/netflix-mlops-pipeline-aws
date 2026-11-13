import sagemaker
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.feature_store.feature_group import FeatureGroup
import pandas as pd
import time

def run_phase_2():
    bucket = "company-mlops-netflix-production-592444418608"
    role_arn = "arn:aws:iam::592444418608:role/AmazonSageMaker-ExecutionRole-NetflixMLOps"
    sess = sagemaker.Session(default_bucket=bucket)

    print("STEP 1: Running SageMaker Processing Job...")
    sklearn_processor = SKLearnProcessor(
        framework_version='1.2-1',
        role=role_arn,
        instance_type='ml.m5.large',
        instance_count=1,
        sagemaker_session=sess
    )

    sklearn_processor.run(
        code='scripts/preprocessing.py',
        inputs=[ProcessingInput(source=f"s3://{bucket}/raw/netflix_titles.csv", destination='/opt/ml/processing/input/')],
        outputs=[ProcessingOutput(source='/opt/ml/processing/output/', destination=f"s3://{bucket}/processed/")]
    )
    print("✅ Processing Job COMPLETED!")

    print("\nSTEP 2: Loading processed data...")
    df = pd.read_csv(f"s3://{bucket}/processed/processed_data.csv")
    print(f"Loaded {len(df)} records.")

    print("\nSTEP 3: Creating Feature Group...")
    feature_group_name = 'netflix-catalog-feature-group'

    # Clean up if exists to prevent schema conflicts
    try:
        fg = FeatureGroup(name=feature_group_name, sagemaker_session=sess)
        fg.delete()
        print("Deleted existing feature group for clean state.")
        time.sleep(10)
    except Exception:
        pass

    # Create new
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=sess)
    feature_group.load_feature_definitions(data_frame=df)

    print("Creating Feature Group...")
    feature_group.create(
        s3_uri=f"s3://{bucket}/offline-store",
        record_identifier_name='record_id',
        event_time_feature_name='event_time',
        role_arn=role_arn,
        enable_online_store=True
    )

    # Wait with TIMEOUT (max 5 minutes)
    print("Waiting for creation (max 5 minutes)...")
    for i in range(30):
        try:
            status = feature_group.describe()['FeatureGroupStatus']
            print(f"Status: {status}")
            if status == 'Created':
                print("✅ Feature Group created successfully!")
                break
            elif status == 'Failed':
                print("❌ Feature Group creation FAILED.")
                break
        except Exception as e:
            print(f"Error checking status: {e}")
            break
        time.sleep(10)

    print("\nSTEP 4: Ingesting data...")
    try:
        # max_workers=1 prevents Windows multiprocessing spawn errors
        feature_group.ingest(data_frame=df, max_workers=1, wait=True)
        print("\n" + "="*50)
        print("🎉 PHASE 2 COMPLETE!")
        print("="*50)
    except Exception as e:
        print(f"\n❌ Ingestion failed: {e}")

# THIS LINE IS MANDATORY FOR WINDOWS TO PREVENT MULTIPROCESSING CRASHES
if __name__ == '__main__':
    run_phase_2()