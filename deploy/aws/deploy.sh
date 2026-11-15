#!/usr/bin/env bash
# Build the image, deploy/update the stack with a content-addressed immutable tag, then push the image.
# Needs: docker, AWS CLI credentials for the account, and the SSM SecureString $FIRMS_KEY_PARAMETER.
set -euo pipefail

STACK=${STACK:-fire-atlas-daily-ingest}
REGION=${AWS_REGION:-us-east-2}
FIRMS_KEY_PARAMETER=${FIRMS_KEY_PARAMETER:-/fire-atlas/firms-map-key}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)

VPC=${VPC_ID:-$(aws ec2 describe-vpcs --region "$REGION" --filters Name=is-default,Values=true \
  --query 'Vpcs[0].VpcId' --output text)}
SUBNETS=${SUBNET_IDS:-$(aws ec2 describe-subnets --region "$REGION" --filters Name=vpc-id,Values="$VPC" \
  Name=default-for-az,Values=true --query 'Subnets[].SubnetId' --output text | tr '\t' ',')}
aws ssm get-parameter --region "$REGION" --name "$FIRMS_KEY_PARAMETER" --query Parameter.Type --output text >/dev/null

docker build -t fire-atlas-data:deploy "$ROOT"
TAG=$(docker image inspect -f '{{.Id}}' fire-atlas-data:deploy | cut -d: -f2 | cut -c1-16)

aws cloudformation deploy --region "$REGION" --stack-name "$STACK" \
  --template-file "$ROOT/deploy/aws/daily-ingest.yaml" --capabilities CAPABILITY_IAM \
  --parameter-overrides ImageTag="$TAG" VpcId="$VPC" SubnetIds="$SUBNETS" FirmsKeyParameterName="$FIRMS_KEY_PARAMETER"

REPO=$(aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
  --query "Stacks[0].Outputs[?OutputKey=='RepositoryUri'].OutputValue" --output text)
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${REPO%%/*}"
if ! aws ecr describe-images --region "$REGION" --repository-name "${REPO#*/}" --image-ids imageTag="$TAG" >/dev/null 2>&1; then
  docker tag fire-atlas-data:deploy "$REPO:$TAG"
  docker push "$REPO:$TAG"
fi
echo "deployed $STACK with image $REPO:$TAG"
