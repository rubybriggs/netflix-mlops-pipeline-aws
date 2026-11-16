Write-Host "=== STARTING FULL CLEANUP ===" -ForegroundColor Yellow

$endpoint = "netflix-mlops-endpoint"
$feature_group = "netflix-catalog-feature-group"

# 1. Delete Endpoint (Stops the ~$30/day billing immediately)
Write-Host "Deleting Endpoint: $endpoint..."
aws sagemaker delete-endpoint --endpoint-name $endpoint --no-cli-pager
Write-Host "Waiting for endpoint deletion (this takes ~5 mins)..."
aws sagemaker wait endpoint-deleted --endpoint-name $endpoint
Write-Host "✅ Endpoint deleted." -ForegroundColor Green

# 2. Delete Feature Group (Stops DynamoDB online store billing)
Write-Host "Deleting Feature Group: $feature_group..."
aws sagemaker delete-feature-group --feature-group-name $feature_group --no-cli-pager
Write-Host "✅ Feature Group deleted." -ForegroundColor Green

Write-Host "=== CLEANUP COMPLETE! BILLING STOPPED. ===" -ForegroundColor Green