import boto3
import sagemaker
import time

def deploy_model():
    print("="*60)
    print(" PHASE 4: DEPLOYING MODEL TO REAL-TIME ENDPOINT")
    print("="*60)
    
    bucket = "company-mlops-netflix-production-592444418608"
    role_arn = "arn:aws:iam::592444418608:role/AmazonSageMaker-ExecutionRole-NetflixMLOps"
    sess = sagemaker.Session(default_bucket=bucket)
    region = sess.boto_region_name
    
    sm_client = boto3.client('sagemaker', region_name=region)
    asg_client = boto3.client('application-autoscaling', region_name=region)
    
    model_package_group_name = "NetflixMLOpsModelGroup"
    endpoint_name = "netflix-mlops-endpoint"
    
    # 1. Find the latest APPROVED model package
    print("\n STEP 1: Fetching latest approved model from Registry...")
    response = sm_client.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1
    )
    
    if not response['ModelPackageSummaryList']:
        print("❌ No approved model found. Please run the approval command first.")
        return
    
    model_package_arn = response['ModelPackageSummaryList'][0]['ModelPackageArn']
    print(f"✅ Found approved model: {model_package_arn}")
    
    # 2. Create SageMaker Model
    print("\n STEP 2: Creating SageMaker Model...")
    model_name = f"netflix-xgb-model-{int(time.time())}"
    sm_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role_arn,
        Containers=[{'ModelPackageName': model_package_arn}]
    )
    print(f"✅ Model created: {model_name}")
    
    # 3. Create Endpoint Config
    print("\n STEP 3: Creating Endpoint Configuration...")
    endpoint_config_name = f"netflix-epc-{int(time.time())}"
    sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            'VariantName': 'AllTraffic',
            'ModelName': model_name,
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.m5.large',
            'InitialVariantWeight': 1.0
        }]
    )
    print(f"✅ Endpoint Config created: {endpoint_config_name}")
    
    # 4. Create Endpoint
    print(f"\n STEP 4: Deploying Endpoint '{endpoint_name}' (This takes ~5-8 mins)...")
    try:
        sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
    except sm_client.exceptions.ResourceInUseException:
        print("Endpoint already exists. Updating...")
        sm_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
    
    # Wait for endpoint
    print("Waiting for endpoint to be InService...")
    waiter = sm_client.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=endpoint_name)
    print("✅ Endpoint is now InService!")
    
    # 5. Auto-Scaling
    print("\n STEP 5: Configuring Auto-Scaling...")
    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"
    
    asg_client.register_scalable_target(
        ServiceNamespace='sagemaker',
        ResourceId=resource_id,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        MinCapacity=1,
        MaxCapacity=3
    )
    
    asg_client.put_scaling_policy(
        PolicyName=f"Invocations-ScalingPolicy-{endpoint_name}",
        ServiceNamespace='sagemaker',
        ResourceId=resource_id,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        PolicyType='TargetTrackingScaling',
        TargetTrackingScalingPolicyConfiguration={
            'TargetValue': 1000.0,
            'PredefinedMetricSpecification': {
                'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
            },
            'ScaleInCooldown': 300,
            'ScaleOutCooldown': 60
        }
    )
    print("✅ Auto-scaling configured!")
    
    print("\n" + "="*60)
    print("🎉 PHASE 4 COMPLETE!")
    print("="*60)

if __name__ == '__main__':
    try:
        deploy_model()
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")