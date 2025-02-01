import sys
import os
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(src_path)

import random
import json

from src.Intent_classification import intent_prediction  # importing intent prediction from intent classification
from src.Entity_extraction import entity_extraction # importing entity extraction function from entityExtraction
# from response_data.response import get_responses  # Import the get_response function from response_data

'''
def generate_response(query):
    # Get intent from the classification model
    intent = intent_prediction(query)  # Assuming this function returns the intent
    response = random.choice(get_responses(intent))  # Get response based on the intent
    return response

# creating the loops to create a conversation
while (user_query := input("You: ").lower()) not in ["quit", "back", "thats it", "bye"]:
    response = generate_response(user_query)
    print(response)
'''


# Load responses from JSON file
def load_responses(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

# Fetch response based on raw user query
def generate_response(user_query):
    file_path = os.path.join(os.path.dirname(__file__), "../response_data/responses.json")
    response_data = load_responses(file_path)
    responses = []
    # Step 1: Predict Intent
    intent = intent_prediction(user_query)
    
    # Step 2: Extract Entities
    entities = entity_extraction(user_query)

    # Step 3: Retrieve Response
    if intent in response_data:
        intent_responses = response_data[intent]

        if not entities:  # If entities list is empty
            return random.choice(intent_responses["default"])

        for entity in entities:
            if entity in intent_responses:
                responses.append(random.choice(intent_responses[entity]))
            else:
                responses.append(random.choice(intent_responses["default"]))
    
        if not responses:
            return random.choice(intent_responses["default"])
    response = "\n".join(responses)
    return response

