#!/bin/bash
set -euo pipefail

PROFILE="mahamedia"
STACK_NAME="signals-dashboard"
REGION="us-east-1"
S3_BUCKET="signals-deploy-artifacts"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Signals Dashboard Deploy ==="

echo "Building frontend..."
cd "$PROJECT_DIR/frontend"
npm run build

echo "Packaging SAM template..."
cd "$SCRIPT_DIR"
sam build --template template.yaml

echo "Deploying..."
sam deploy \
  --profile "$PROFILE" \
  --stack-name "$STACK_NAME" \
  --region "$REGION" \
  --s3-bucket "$S3_BUCKET" \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    DomainName=trader.mahamedia.us \
    HostedZoneId=ZXXXXXXXXXXXXX \
    CertificateArn=arn:aws:acm:us-east-1:XXXX:certificate/XXXX \
  --no-confirm-changeset

echo "Done! https://trader.mahamedia.us"
