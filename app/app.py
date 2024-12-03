import os
import requests
import json
from flask import Flask, request, render_template, jsonify
import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.session import Session

app = Flask(__name__)

# Configuration for Bedrock API
bedrock_region = "us-west-2"
bedrock_model_id = "amazon.titan-text-lite-v1"
bedrock_endpoint = "https://bedrock-runtime.us-west-2.amazonaws.com"

# Function to create SigV4 signed request headers for Bedrock API
def get_bedrock_headers(payload):
    session = Session()
    credentials = session.get_credentials()
    
    # Check if credentials are loaded
    if credentials is None:
        raise Exception("AWS credentials not found. Ensure they are configured properly.")
    
    # Create AWSRequest for SigV4 signing
    request = AWSRequest(
        method="POST",
        url=f"{bedrock_endpoint}/model/{bedrock_model_id}/invoke",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"}
    )
    
    SigV4Auth(credentials, "bedrock", bedrock_region).add_auth(request)
    return dict(request.headers)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate_text():
    input_text = request.form['input_text']
    
    payload_body = {
        "inputText": input_text,
        "textGenerationConfig": {
            "maxTokenCount": 500,
            "stopSequences": [],
            "temperature": 0.7,
            "topP": 1
        }
    }
    
    payload = json.dumps(payload_body)

    try:
        headers = get_bedrock_headers(payload)
        
        response = requests.post(
            f"{bedrock_endpoint}/model/{bedrock_model_id}/invoke",
            headers=headers,
            data=payload
        )

        response_payload = response.json()
        results = response_payload.get("results", [])
        if results:
            output_text = results[0].get("outputText", "No output generated.")
        else:
            output_text = "No results available."
        print(f"{output_text} is output")
        # return jsonify({"input_text": input_text, "generated_text": output_text})
        return render_template('response.html', input_text=input_text, generated_text=output_text)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
