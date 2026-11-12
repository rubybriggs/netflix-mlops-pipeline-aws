import boto3
import logging
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Configuration
REGION = 'us-east-1'
ACCOUNT_ID = boto3.client('sts', region_name=REGION).get_caller_identity().get('Account')
BUCKET_NAME = f"company-mlops-netflix-production-{ACCOUNT_ID}"

s3_client = boto3.client('s3', region_name=REGION)

def create_secure_bucket():
    try:
        # Check if bucket already exists
        s3_client.head_bucket(Bucket=BUCKET_NAME)
        logger.info(f"Bucket '{BUCKET_NAME}' already exists. Skipping creation.")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            logger.info(f"Creating bucket: {BUCKET_NAME}")
            try:
                if REGION == 'us-east-1':
                    s3_client.create_bucket(Bucket=BUCKET_NAME)
                else:
                    s3_client.create_bucket(
                        Bucket=BUCKET_NAME,
                        CreateBucketConfiguration={'LocationConstraint': REGION}
                    )
                
                # Enforce SSE-KMS encryption at the bucket level
                s3_client.put_bucket_encryption(
                    Bucket=BUCKET_NAME,
                    ServerSideEncryptionConfiguration={
                        'Rules': [
                            {
                                'ApplyServerSideEncryptionByDefault': {
                                    'SSEAlgorithm': 'aws:kms'
                                },
                                'BucketKeyEnabled': True
                            }
                        ]
                    }
                )
                logger.info(f"Successfully created bucket '{BUCKET_NAME}' with SSE-KMS encryption.")
            except Exception as err:
                logger.error(f"Failed to create bucket: {err}")
                raise
        else:
            logger.error(f"Unexpected error checking bucket: {e}")
            raise

if __name__ == "__main__":
    create_secure_bucket()
    logger.info("S3 Setup Complete. Prefixes (/raw, /processed, /artifacts, /pipeline_root) will be created dynamically upon first upload.")