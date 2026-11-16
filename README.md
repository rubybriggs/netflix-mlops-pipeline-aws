\# 🎬 Netflix MLOps Pipeline on AWS



> End-to-end machine learning operations pipeline for the Netflix catalog dataset —

> from raw CSV ingestion to a live, auto-scaling real-time inference endpoint, fully

> orchestrated as a SageMaker Pipeline DAG and version-controlled in Git.



!\[Status](https://img.shields.io/badge/Phase\_1–4-✅\_Complete-success)

!\[Region](https://img.shields.io/badge/Region-us--east--1-blue)

!\[Framework](https://img.shields.io/badge/SageMaker\_SDK-v2-orange)

!\[Model](https://img.shields.io/badge/Algorithm-XGBoost-green)



\---



\## 📖 Overview



This project implements a complete MLOps lifecycle for a \*\*content-based classification /

similarity task\*\* on the Netflix titles dataset (8,807 records). It demonstrates:



\- \*\*Feature engineering\*\* with TF-IDF semantic text embeddings (100 features) + date features

\- \*\*SageMaker Feature Store\*\* with both offline (S3 + Glue Data Catalog) and online (DynamoDB) stores

\- \*\*SageMaker Pipelines\*\* — a 5-step DAG: Process → Train → Evaluate → Condition → Register

\- \*\*Model governance\*\* via a quality gate (accuracy ≥ 0.85) and manual approval in the Model Registry

\- \*\*Real-time deployment\*\* with target-tracking auto-scaling and verified live inference



\### Architecture



```mermaid

flowchart LR

&#x20;   A\[Raw CSV\\ns3://.../raw] --> B\[ProcessData\\nSKLearn Processing Job]

&#x20;   B --> C\[Feature Store\\nOffline S3/Glue + Online DynamoDB]

&#x20;   B --> D\[TrainModel\\nXGBoost built-in]

&#x20;   D --> E\[EvaluateModel\\naccuracy metric]

&#x20;   E --> F{CheckAccuracy\\n≥ 0.85?}

&#x20;   F -->|Yes| G\[RegisterApprovedModel\\nModel Registry]

&#x20;   F -->|No| H\[Pipeline Fails\\nno registration]

&#x20;   G --> I\[Deploy Endpoint\\nml.m5.large + auto-scale]

&#x20;   I --> J\[Live Inference]





🏗️ Project Structure

text

aws-netflix-mlops-pipeline/

├── config/

│   ├── featurestore-s3-policy.json   # s3:GetBucketAcl / PutBucketAcl inline policy

│   └── deploy-policy.json            # endpoint + auto-scaling permissions

├── scripts/

│   ├── preprocessing.py              # Phase 2: TF-IDF embeddings + Unicode sanitization

│   ├── run\_phase2.py                 # Phase 2 orchestrator (processing + feature store)

│   ├── train\_knn.py                  # Phase 3A standalone K-NN trainer (reference)

│   ├── run\_phase3a.py                # Phase 3A standalone training (numeric-only)

│   ├── evaluate.py                   # Phase 3: accuracy evaluation → evaluation.json

│   ├── build\_pipeline.py             # Phase 3/4: full DAG definition + execution

│   ├── deploy\_endpoint.py            # Phase 4: registry → endpoint + auto-scaling

│   ├── test\_endpoint.py              # Phase 4: live inference verification

│   └── standalone\_phase4.py          # Phase 4 helper

├── README.md

└── requirements.txt                  # (recommended — see Setup)



⚙️ Configuration

Resource identifiers are currently hardcoded in the scripts. For a production-grade

repo, replace them with environment variables or a config/settings.py. The values used:

Variable

Value

AWS\_ACCOUNT\_ID

592444418608

AWS\_REGION

us-east-1

S3\_BUCKET

company-mlops-netflix-production-592444418608

IAM\_ROLE

arn:aws:iam::592444418608:role/AmazonSageMaker-ExecutionRole-NetflixMLOps

FEATURE\_GROUP

netflix-catalog-feature-group

PIPELINE\_NAME

NetflixMLOpsPipeline

MODEL\_PACKAGE\_GROUP

NetflixMLOpsModelGroup

ENDPOINT\_NAME

netflix-mlops-endpoint

ACCURACY\_THRESHOLD

0.85

Recommended cleanup: move these into config/settings.py and import them in every

script so the repo is portable across accounts/regions without editing code.



🚀 Setup

Prerequisites

Python 3.9+ (Anaconda base env works)

AWS CLI v2 configured (aws configure) with credentials for the account above

An S3 bucket with the raw dataset uploaded to s3://<bucket>/raw/netflix\_titles.csv



Install dependencies

Create requirements.txt (the project never generated one — add this for reproducibility):

sagemaker>=2.200.0

pandas

scikit-learn

xgboost

boto3

joblib



Then:



pip install -r requirements.txt



Grant required IAM permissions

The execution role needs two inline policies beyond the managed SageMaker policies.

Apply them once (see config/\*.json):



powershell

aws iam put-role-policy --role-name AmazonSageMaker-ExecutionRole-NetflixMLOps `

&#x20; --policy-name FeatureStoreS3BucketAclAccess `

&#x20; --policy-document file://config/featurestore-s3-policy.json



aws iam put-role-policy --role-name AmazonSageMaker-ExecutionRole-NetflixMLOps `

&#x20; --policy-name SageMakerDeploymentAccess `

&#x20; --policy-document file://config/deploy-policy.json



⚠️ The FeatureStoreS3BucketAclAccess policy is mandatory for custom-named buckets.

The AWS managed AmazonSageMakerFeatureStoreAccess policy only grants s3:GetBucketAcl

to buckets containing the word "SageMaker" — see Troubleshooting below.







▶️ Running the Pipeline

Execute the phases in order. Each is idempotent and safe to re-run.

Phase 2 — Feature Engineering \& Feature Store



powershell

python scripts/run\_phase2.py



Runs a SageMaker Processing Job (TF-IDF embeddings + sanitization), then creates the

Feature Group and ingests all 8,807 records into offline + online stores.



Phase 3 — DAG Orchestration (Train → Evaluate → Condition → Register)

powershell

python scripts/build\_pipeline.py



Defines and starts the NetflixMLOpsPipeline execution. This runs asynchronously in

the cloud (\~10–15 min). Do not proceed until the status is Succeeded:

powershell

aws sagemaker list-pipeline-executions --pipeline-name NetflixMLOpsPipeline `

&#x20; --max-results 1 --query 'PipelineExecutionSummaries\[0].PipelineExecutionStatus' --output text



Phase 4 — Approve \& Deploy

Once the pipeline succeeds, the model lands in the registry as PendingManualApproval.

Approve it (CLI bypasses the console menu):

powershell

$ModelArn = aws sagemaker list-model-packages --model-package-group-name NetflixMLOpsModelGroup `

&#x20; --model-approval-status PendingManualApproval --sort-by CreationTime --sort-order Descending `

&#x20; --query 'ModelPackageSummaryList\[0].ModelPackageArn' --output text

aws sagemaker update-model-package --model-package-arn $ModelArn --model-approval-status Approved

Then deploy the live endpoint with auto-scaling:

powershell

python scripts/deploy\_endpoint.py

Verify live inference

powershell

python scripts/test\_endpoint.py

Expected output:

text

PREDICTION SUCCESSFUL!

Probability of being a Movie: 0.5068

Interpretation: MOVIE (threshold 0.5)





📊 Model \& Evaluation

Item

Detail

Algorithm

XGBoost (SageMaker built-in, 1.7-1)

Task

Binary classification — Movie (1) vs TV Show (0)

Features

100 TF-IDF embeddings + release\_year (101 numeric)

Target

type mapped to 0/1

Quality gate

ConditionGreaterThanOrEqualTo(accuracy ≥ 0.85)

Endpoint

ml.m5.large, 1 instance, auto-scaled 1–3

Scaling policy

Target-tracking on SageMakerVariantInvocationsPerInstance = 1000

Note on the gate: if the model's accuracy falls below 0.85, the Condition Step

correctly refuses to register the model and the pipeline fails. This is intended

governance behavior, not a bug. Lower ACCURACY\_THRESHOLD in build\_pipeline.py only

if the real metric justifies it.





🧯 Troubleshooting \& Lessons Learned

These are the real failure modes encountered and resolved during this build. Each fix is

already baked into the scripts.

Symptom

Root Cause

Fix

CreateFeatureGroup → s3:GetBucketAcl ... AccessDenied

Managed policy only covers buckets named with "SageMaker"

Add inline policy granting s3:GetBucketAcl/PutBucketAcl on the specific bucket

Processing job crashes silently

No error capture in container

Wrap scripts in try/except → write error\_log.txt to S3 output

TypeError: "value" must be scalar/dict/Series, passed "Index"

df.index.astype(str) returns Index, not Series

Use df.index.to\_series().astype(str)

freeze\_support() / multiprocessing crash on Windows

feature\_group.ingest() spawns processes; Windows needs a main guard

Wrap entry in if \_\_name\_\_ == '\_\_main\_\_': and use max\_workers=1

PutRecord ... regex pattern .\* validation error

Feature Store rejects Unicode (em-dash, smart quotes)

Sanitize all text to ASCII + strip newlines before ingestion

K-NN Found entries with missing values

Built-in K-NN requires numeric-only features

Drop text columns; keep only TF-IDF + numeric year; map label to 0/1

Saved ... with 1 numeric features

TF-IDF block was stripped from preprocessing

Generate embeddings after sanitization; verify column count = 114

ImportError: cannot import name 'ConditionStep'

SDK v2 puts it in its own module

from sagemaker.workflow.condition\_step import ConditionStep

'PropertyFile' object is not subscriptable

Can't index a PropertyFile directly

Use JsonGet(step\_name=..., property\_file=..., json\_path=...)

Pipeline variables do not support \_\_str\_\_

f-strings can't embed pipeline properties

Use Join(on="/", values=\[...]) for runtime concatenation

'list' object has no attribute '\_to\_request\_dict'

model\_metrics passed as a list

Pass a single ModelMetrics(...) object, not \[...]

XGBoost training fails instantly, no artifact

Missing output\_path on the Estimator

Set output\_path=f"s3://{bucket}/artifacts"

evaluate.py FileNotFoundError

Script looked for test.csv, data uploaded as train.csv

Match the filename to what was uploaded

update-model-package → Value 'None'

Pipeline still Executing when approval ran

Wait for status Succeeded before approving



💰 Cost \& Cleanup

All compute is billed while running. Shut resources down when not actively working:

powershell

\# Delete the real-time endpoint (largest ongoing cost)

aws sagemaker delete-endpoint --endpoint-name netflix-mlops-endpoint

aws sagemaker delete-endpoint-config --endpoint-config-name netflix-epc-<timestamp>



\# Delete the Feature Group (stops DynamoDB online-store billing)

aws sagemaker delete-feature-group --feature-group-name netflix-catalog-feature-group



\# Delete the pipeline definition (optional)

aws sagemaker delete-pipeline --pipeline-name NetflixMLOpsPipeline





🗺️ Roadmap — Phase 5 (CI/CD \& Monitoring)

CI/CD: GitHub Action / CodePipeline to auto-trigger build\_pipeline.py on push to main

Model Monitoring: SageMaker Model Monitor for data drift + quality on the live endpoint

Parameterization: move hardcoded IDs into config/settings.py / env vars

Blue/Green deployment: shadow tests + traffic shifting on endpoint updates

Drift alerting: CloudWatch alarms on InvocationsPerInstance and monitor metrics



👤 Author

Ruby Briggs · rubybriggs/netflix-mlops-pipeline-aws



📄 License

Internal educational / portfolio project. All rights reserved.

