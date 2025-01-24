
# Semantic Similṁrity For Intent Classification
# model = Fine Tuned Sentence-transformers

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine


def intent_classifier(query):
    # load the Fine Tuned Model
    # create a instance of model
    model = SentenceTransformer("models/Fine_Tuned_Intent_Model")

    # Example FAQs or intents
    # string representation of the Question
    faqs = ["Tell me about the College.",
            "What is the admission process?",
            "What are the facilities available here?",
            "Who is Head of Department Of SSD?",
            "How many Departments do you have?",
            "What are the fees?",
            "What are the placement opportunities?",
            "Does the college have a strong alumni network?",
            "What kind of extracurricular activities are available?"
            "Does the college support research?",
            "Does the college have international partnerships?",
            "Are online courses available?"
            ]

    # Encode FAQs
    # turning the FAQs into vector ( or numerical form which is also know as embedings)
    faq_embeddings = model.encode(faqs, normalize_embeddings=True)

    # Create a mapping of embeddings to FAQ labels
    # creating a dictionary representation of the FAQs : Embedings
    # zip() function combines the FAQs list and FAQ embeding list into pairs.
    faq_to_embedding = {faq: embedding for faq, embedding in zip(faqs, faq_embeddings)}  # one line for loop

    # Encode the query
    # Converting the User Question into vector form (or Numerical form also known as embedings)
    query_embedding = model.encode(query, normalize_embeddings=True)

    # intializing two variables to find the shortest distance
    closest_faq = None
    min_distance = float("inf")

    # loops to find the Closest FAQs to the User Query
    # comparing all the FAQs and user query Embedings
    for faq, embedding in faq_to_embedding.items():
        distance = cosine(query_embedding, embedding)
        if distance < min_distance:
            min_distance = distance
            if min_distance > 0.6:
                closest_faq = "what is this?"
                break
            closest_faq = faq

    print(f"User Query: {query}")
    # printing the closest FAQ
    # output shortest distance FAQs
    #print(f"Closest FAQ: {closest_faq}")

    # FAQs and intents are in dictionary form
    # Each intent is connected to a query
    faq_to_intent = {
        "Tell me about the College.": "COLLEGE_OVERVIEW",
        "What courses are offered?": "COURSE_QUERY",
        "What is the admission process?": "ADMISSION_QUERY",
        "What are the facilities available here?": "CAMPUS_QUERY",
        "Who is Head of Department Of SSD?": "FACULTY_QUERY",
        "How many Departments do you have?": "DEPARTMENT_QUERY",
        "What are the fees?": "FEE_QUERY",
        "What are the placement opportunities?":"PLACEMENT_AND_CAREER_QUERY",
        "Does the college have a strong alumni network?": "ALUMNI_NETWORK_QUERY",
        "What kind of extracurricular activities are available?":"EVENT_AND_ACTIVITES_QUERY",
        "Does the college support research?": "RESEARCH_OPPORTUNITIES_QUERY",
        "Does the college have international partnerships?":"INTERNATIONAL_COLLABORATION_QUERY",
        "Are online courses available?":"ONLINE_LEARNING_AND_RESOURCES_QUERY",
        "what is this?":"FALLBACK"
    }

    # Get the matched intent
    matched_intent = faq_to_intent[closest_faq]
    print(f"Matched Intent: {matched_intent}")


# asking user questions repeatedly to identify the intent
while True:
    user_query = input("You: ")
    if user_query.lower() in ["quit", "back", "thats it", "bye"]:
        break
    else:
        intent_classifier(user_query)

















# ***********this is optional******************

# some pre-written response connected to the Intents
# when the intent is defined it chooses the relative response
intent_to_response = {
    "ADMISSION_QUERY": "The admission process involves filling out an online form.",
    "FEE_QUERY": "The fee structure depends on the course. Please specify the course.",
    "COURSE_QUERY": "We offer courses in Engineering, Science, and Arts."
}


#response = intent_to_response[matched_intent]
#print(f"Response: {response}")

# **************this is optional****************
