import sagemaker
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput

# 1. Initialize Session
bucket = "company-mlops-netflix-production-592444418608"
role_arn = "arn:aws:iam::592444418608:role/AmazonSageMaker-ExecutionRole-NetflixMLOps"
sess = sagemaker.Session(default_bucket=bucket)

print("Starting Bare Minimum Test Job...")

# 2. Setup Processor
test_processor = SKLearnProcessor(
    framework_version='1.2-1',
    role=role_arn,
    instance_type='ml.m5.large',
    instance_count=1,
    sagemaker_session=sess
)

# 3. Run the Test
test_processor.run(
    code='scripts/test_bare_minimum.py',
    inputs=[
        ProcessingInput(
            source=f"s3://{bucket}/raw/netflix_titles.csv",
            destination='/opt/ml/processing/input/'
        )
    ]
)
print("✅ Test Job Completed!")