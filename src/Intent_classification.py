# Semantic Similarity For Intent Classification
# model = Fine Tuned Sentence-transformers

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine


def intent_prediction(query):
    # load the Fine Tuned Model
    # create a instance of model
    model = SentenceTransformer("models/Fine_Tuned_Intent_Model")

    # Example FAQs or intents
    # string representation of the Question
    faqs = ["Tell me about the College.",
            "Do you offer evening or weekend classes?",
            "What are the key deadlines for applying?",
            "What are the facilities available here?",
            "Who is Head of Department Of SSD?",
            "Is there a business school or management department?",
            "How much does a program cost?",
            "What is the average placement percentage?",
            "Who are some notable alumni?",
            "Do you organize annual events?",
            "Are there any research labs available?",
            "Can I participate in student exchange programs?",
            "Do you offer online certification programs?",
            "What is the schedule for semester exams at BFGI?",
            "Hi",
            "What courses should I take next semester?"
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

    #print(f"User Query: {query}")
    # printing the closest FAQ
    # output shortest distance FAQs
    #print(f"Closest FAQ: {closest_faq}")

    # FAQs and intents are in dictionary form
    # Each intent is connected to a query
    faq_to_intent = {
        "Tell me about the College.": "COLLEGE_OVERVIEW",
        "Do you offer evening or weekend classes?": "COURSE_QUERY",
        "What are the key deadlines for applying?": "ADMISSION_QUERY",
        "What are the facilities available here?": "CAMPUS_QUERY",
        "Who is Head of Department Of SSD?": "FACULTY_QUERY",
        "Is there a business school or management department?": "DEPARTMENT_QUERY",
        "How much does a program cost?": "FEE_QUERY",
        "What is the average placement percentage?":"PLACEMENT_AND_CAREER_QUERY",
        "Who are some notable alumni?": "ALUMNI_NETWORK_QUERY",
        "Do you organize annual events?":"EVENT_AND_ACTIVITES_QUERY",
        "Are there any research labs available?": "RESEARCH_OPPORTUNITIES_QUERY",
        "Can I participate in student exchange programs?":"INTERNATIONAL_COLLABORATION_QUERY",
        "Do you offer online certification programs?":"ONLINE_LEARNING_AND_RESOURCES_QUERY",
        "What is the schedule for semester exams at BFGI?":"EXAMS_QUERY",
        "Hi" : "GREETING",
        "What courses should I take next semester?":"ACADEMIC_ADVISING",
        "what is this?":"FALLBACK"
    }

    # Get the matched intent
    matched_intent = faq_to_intent[closest_faq]
    #print(f"Matched Intent: {matched_intent}")
    return matched_intent

# asking user questions repeatedly to identify the intent
if __name__ == "__main__": 
    while (user_query := input("You: ").lower()) not in ["quit", "back", "thats it", "bye"]:
        intent_prediction(user_query)


















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
