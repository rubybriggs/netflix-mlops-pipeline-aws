import boto3
import numpy as np

client = boto3.client("sagemaker-runtime", region_name="us-east-1")
endpoint_name = "netflix-mlops-endpoint"

# 100 TF-IDF features + 1 release year, sent as headerless CSV (matches XGBoost)
payload = ",".join(str(x) for x in (np.random.rand(100).tolist() + [2023]))

response = client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="text/csv",
    Body=payload,
)

result = float(response["Body"].read().decode())

print("PREDICTION SUCCESSFUL!")
print("Probability of being a Movie: {:.4f}".format(result))
if result >= 0.5:
    print("Interpretation: MOVIE (threshold 0.5)")
else:
    print("Interpretation: TV SHOW (threshold 0.5)")