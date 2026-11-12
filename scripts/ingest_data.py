import os
import boto3
import pandas as pd
import awswrangler as wr
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Configuration
REGION = 'us-east-1'
ACCOUNT_ID = boto3.client('sts', region_name=REGION).get_caller_identity().get('Account')
BUCKET_NAME = f"company-mlops-netflix-production-{ACCOUNT_ID}"

# Dynamically find the project root folder (one level up from the 'scripts' folder)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
LOCAL_FILE_PATH = os.path.join(project_root, "netflix_titles.csv")
S3_RAW_PATH = f"s3://{BUCKET_NAME}/raw/netflix_titles.csv"

REQUIRED_COLUMNS = [
    'show_id', 'type', 'title', 'director', 'cast', 
    'country', 'date_added', 'release_year', 'rating', 
    'duration', 'listed_in', 'description'
]

def validate_and_ingest():
    if not os.path.exists(LOCAL_FILE_PATH):
        logger.error(f"CRITICAL: File not found at '{LOCAL_FILE_PATH}'. Please verify the path.")
        return False

    try:
        logger.info("Reading and validating local CSV...")
        df = pd.read_csv(LOCAL_FILE_PATH, on_bad_lines='warn', dtype=str)
        
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            logger.error(f"Schema validation FAILED. Missing columns: {missing_cols}")
            return False
        
        logger.info(f"Schema validation PASSED. Found {len(df)} records and all required columns.")

        df = df.fillna("")
        df['description'] = df['description'].str.strip()
        df['cast'] = df['cast'].str.strip()

        logger.info(f"Uploading validated data to {S3_RAW_PATH} with SSE-KMS...")
        wr.s3.to_csv(
            df=df,
            path=S3_RAW_PATH,
            index=False,
            dataset=False,
            s3_additional_kwargs={
                'ServerSideEncryption': 'aws:kms',
                'SSEKMSKeyId': 'alias/aws/s3'
            }
        )
        
        logger.info("✅ Ingestion completed successfully. Data is ready for Phase 2.")
        return True

    except pd.errors.EmptyDataError:
        logger.error("The CSV file is empty.")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during ingestion: {str(e)}")
        return False

if __name__ == "__main__":
    validate_and_ingest()