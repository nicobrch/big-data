import requests
import pandas as pd
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load the validation data
data_validation = pd.read_csv('data/twitter_validation.csv')
data_validation.columns = ['id', 'hilo', 'class', 'tweet']
tweets = data_validation['tweet'].fillna('').tolist()

# Define the endpoints
svm_url = 'http://localhost:5000/svm'
lr_url = 'http://localhost:5000/lr'

# Function to send a POST request
def send_request(url, tweet):
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({'text': tweet})
    response = requests.post(url, headers=headers, data=data)
    return response.json()

# Function to handle requests with threading
def handle_requests(url, tweets):
    responses = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_tweet = {executor.submit(send_request, url, tweet): tweet for tweet in tweets}
        for future in as_completed(future_to_tweet):
            try:
                response = future.result()
                responses.append(response)
            except Exception as exc:
                print(f'Tweet generated an exception: {exc}')
    return responses

# Send requests to the SVM and LR endpoints using threading
svm_start_time = time.time()
svm_responses = handle_requests(svm_url, tweets)
svm_end_time = time.time()

print(f"Total time for {len(tweets)} SVM requests: {svm_end_time - svm_start_time:.4f} seconds")

lr_start_time = time.time()
lr_responses = handle_requests(lr_url, tweets)
lr_end_time = time.time()

print(f"Total time for {len(tweets)} LR requests: {lr_end_time - lr_start_time:.4f} seconds")

# Optionally, save the responses to a file
with open('./metrics/svm_responses.json', 'w') as f:
    json.dump(svm_responses, f, indent=4)

with open('./metrics/lr_responses.json', 'w') as f:
    json.dump(lr_responses, f, indent=4)