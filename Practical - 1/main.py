import boto3
import json
from botocore.exceptions import ClientError

# Initialize Bedrock Runtime Client
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Define the Model ID
model_id = "meta.llama3-8b-instruct-v1:0"


# Define the Prompt
prompt = "Best places to visit in Lucknow"

# Request Payload (Meta Llama 3 does not require special formatting)
native_request = {
    "prompt": prompt,
    "max_gen_len": 512,
    "temperature": 0.5,
}

# Convert request to JSON
request_body = json.dumps(native_request)

try:
    # Invoke Model
    response = client.invoke_model(modelId=model_id, body=request_body)

    # Read and Parse Response
    response_body = json.loads(response["body"].read().decode("utf-8"))

    # Debug: Print Full Response to Check Structure
    print("Full Response JSON:", json.dumps(response_body, indent=4))

except ClientError as e:
    print(f"ERROR: AWS Client Error - {e.response['Error']['Message']}")
except Exception as e:
    print(f"ERROR: {e}")
