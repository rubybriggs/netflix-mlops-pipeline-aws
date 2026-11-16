import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

def build_and_run_pipeline():
    bucket = "company-mlops-netflix-production-592444418608"
    role_arn = "arn:aws:iam::592444418608:role/AmazonSageMaker-ExecutionRole-NetflixMLOps"
    sess = sagemaker.Session(default_bucket=bucket)

    proc = SKLearnProcessor(
        framework_version="1.2-1", role=role_arn,
        instance_type="ml.m5.large", instance_count=1, sagemaker_session=sess
    )
    step_process = ProcessingStep(
        name="BisectProcess", processor=proc,
        inputs=[ProcessingInput(source=f"s3://{bucket}/raw/netflix_titles.csv", destination="/opt/ml/processing/input/")],
        outputs=[ProcessingOutput(source="/opt/ml/processing/output/", destination=f"s3://{bucket}/processed/")],
        code="scripts/preprocessing.py"
    )

    pipeline = Pipeline(name="NetflixMLOpsPipeline-Bisect", steps=[step_process], sagemaker_session=sess)
    pipeline.upsert(role_arn=role_arn)
    ex = pipeline.start()
    print("✅ Bisect Execution Started:", ex.arn)

if __name__ == '__main__':
    try:
        build_and_run_pipeline()
    except Exception as e:
        print("❌ Build failed:", e)