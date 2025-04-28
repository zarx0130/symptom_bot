# for temporary code


# temp

"""
from flask import Flask, request, jsonify, render_template
import pandas as pd
from collections import defaultdict
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from sentence_transformers import SentenceTransformer  # For better semantic matching

app = Flask(__name__)

# Load models
nlp = spacy.load("en_core_web_sm")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight sentence transformer

# Cache for faster responses
data_cache = None
vectorizer_cache = None
symptom_vectors_cache = None
symptom_embeddings_cache = None

def load_data(file):
    global data_cache
    if data_cache is None:
        data = pd.read_csv(file)
        data['Symptoms_processed'] = data['Symptoms'].apply(preprocess_text)
        data['symptom_embeddings'] = list(sentence_model.encode(data['Symptoms_processed'].tolist()))
        data_cache = data
    return data_cache

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def get_vectorizer(data):
    global vectorizer_cache, symptom_vectors_cache
    if vectorizer_cache is None:
        vectorizer = TfidfVectorizer()
        symptom_vectors = vectorizer.fit_transform(data['Symptoms_processed'])
        vectorizer_cache = vectorizer
        symptom_vectors_cache = symptom_vectors
    return vectorizer_cache, symptom_vectors_cache

def match_symptoms(user_input, data, threshold=0.5):
    # Try both TF-IDF and sentence embeddings for better coverage
    user_input_processed = preprocess_text(user_input)
    
    # Method 1: TF-IDF matching
    vectorizer, symptom_vectors = get_vectorizer(data)
    user_vector = vectorizer.transform([user_input_processed])
    tfidf_similarities = cosine_similarity(user_vector, symptom_vectors)
    tfidf_indices = np.where(tfidf_similarities > threshold)[1]
    
    # Method 2: Sentence embedding matching
    user_embedding = sentence_model.encode([user_input_processed])
    embedding_similarities = cosine_similarity(user_embedding, np.stack(data['symptom_embeddings']))
    embedding_indices = np.where(embedding_similarities > threshold)[1]
    
    # Combine results
    combined_indices = set(tfidf_indices).union(set(embedding_indices))
    matched_symptoms = data.iloc[list(combined_indices)]['Symptoms'].tolist()
    
    return matched_symptoms

def predict_illnesses(matched_symptoms, data):
    illnesses = defaultdict(int)
    symptom_scores = defaultdict(float)
    
    for symptom in matched_symptoms:
        matches = data[data['Symptoms'] == symptom]
        if not matches.empty:
            for _, row in matches.iterrows():
                illnesses[row['Illness']] += 1
                symptom_scores[row['Illness']] += max(
                    cosine_similarity(
                        sentence_model.encode([preprocess_text(symptom)]),
                        [row['symptom_embeddings']]
                    )[0][0]
                )
    
    # Combine count and similarity score
    scored_illnesses = []
    for illness, count in illnesses.items():
        avg_score = symptom_scores[illness] / count
        scored_illnesses.append((illness, count, avg_score))
    
    # Sort by score then count
    scored_illnesses.sort(key=lambda x: (-x[2], -x[1]))
    return scored_illnesses

def format_response(illness_info):
    response = f"**{illness_info['Illness']}**\n"
    response += f"- Symptoms: {illness_info['Symptoms']}\n"
    response += f"- Severity: {illness_info['Severity']}\n"
    if illness_info['Contagious'] != 'No':
        response += f"- Contagious: {illness_info['Contagious']}\n"
    response += f"- Treatment: {illness_info['Treatment']}\n"
    if pd.notna(illness_info['Notes']):
        response += f"- Notes: {illness_info['Notes']}\n"
    return response

def chatbot(user_input):
    try:
        data = load_data('sx.csv')
        matched_symptoms = match_symptoms(user_input, data)
        
        if not matched_symptoms:
            return "I couldn't match your symptoms to known conditions. Could you describe them differently? For example: 'I have a headache and fever'."
        
        predicted_illnesses = predict_illnesses(matched_symptoms, data)
        
        if not predicted_illnesses:
            return "No clear matches found. Please consult a healthcare professional."
        
        result_msg = "Based on your symptoms, here are possible conditions (sorted by likelihood):\n\n"
        
        # Top 3 matches with details
        for illness, count, score in predicted_illnesses[:3]:
            illness_info = data[data['Illness'] == illness].iloc[0]
            result_msg += format_response(illness_info)
            result_msg += f"(Matched {count} symptoms, confidence: {score:.2f})\n\n"
        
        # Additional matches if any
        if len(predicted_illnesses) > 3:
            result_msg += "Other possible conditions:\n"
            others = [illness for illness, _, _ in predicted_illnesses[3:6]]
            result_msg += ", ".join(others) + "\n"
        
        result_msg += "\nNote: This is not a diagnosis. Please consult a doctor for medical advice."
        return result_msg
    
    except Exception as e:
        print(f"Error processing request: {e}")
        return "Sorry, I encountered an error processing your request. Please try again."

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = request.get_json()
        user_input = data.get('symptoms', '')
        if not user_input.strip():
            return jsonify({'result': "Please describe your symptoms."})
        result_msg = chatbot(user_input)
        return jsonify({'result': result_msg})
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify, render_template
import pandas as pd # type: ignore
from collections import defaultdict
import difflib
import re

app = Flask(__name__)

# load dataset
def load_data(file):
    data = pd.read_csv(file) # sx.csv
    return data

# symptom-illness map 
def build_map(data):
    map = defaultdict(list)
    known_symptoms = set() 
    for index, row in data.iterrows():
        symptoms = row['Symptoms'].split(',')
        illness_info = {
            'illness': row['Illness'],
            'symptoms': row["Symptoms"],
            'severity': row['Severity'],
            'contagious': row['Contagious'],
            'treatment': row['Treatment'],
            'notes': row['Notes']
        }
        for symptom in symptoms:
            symptom = symptom.strip().lower()
            map[symptom].append(illness_info)
            known_symptoms.add(symptom)
    return map, known_symptoms

# extract symptoms from user input
def extract_sx(user_input, known_symptoms):
    extracted_symptoms = []
    for symptom in known_symptoms:
        if re.search(r'\b' + re.escape(symptom) + r'\b', user_input.lower()):
            extracted_symptoms.append(symptom)
    return extracted_symptoms
        
# match user-provided symptoms to known symptoms (w/ sp tolerance)
def match_sx(user_sx, map):
    matches = []
    for symptom in user_sx:
        match = difflib.get_close_matches(symptom, map.keys(), n=1, cutoff=0.8)
        if match:
            matches.append(match[0])
    return matches

# predict illness based on matched symptoms
def predict(matches, map):
    illnesses = defaultdict(int)
    for match in matches:
        for illness_info in map[match]:
            illness = illness_info['illness']
            illnesses[illness] += 1
    return illnesses

# chatbot function
def chatbot(user_input):
    # load dataset
    data = load_data('sx.csv')
    # build symptom-illness map
    map, known_symptoms = build_map(data)
    # extract symptoms from user input
    user_sx = extract_sx(user_input, known_symptoms)
    #print(f"Extracted symptoms: {user_sx}")  # Debug print
    if not user_sx:
        return 'No symptoms detected, please try again.'

    # match symptoms
    matches = match_sx(user_sx, map)
    # predict illness
    illnesses = predict(matches, map)
    # sort illnesses by the number of matched symptoms
    sorted_illnesses = sorted(illnesses.items(), key=lambda x: x[1], reverse=True)
    # prepare the result message
    result_msg = ""
    if sorted_illnesses:
        result_msg += 'Here are the predicted illnesses based on the provided symptoms:\n\n'
        most_likely_illness = sorted_illnesses[0]
        result_msg += f'Most likely illness: \n {most_likely_illness[0]} \n'
        printed_illnesses = set()
        for match in matches:
            for illness_info in map[match]:
                if illness_info['illness'] == most_likely_illness[0] and illness_info['illness'] not in printed_illnesses:
                    result_msg += f"  <strong>Symptoms</strong>: {illness_info['symptoms']}\n"
                    result_msg += f"  <strong>Severity</strong>: {illness_info['severity']}\n"
                    result_msg += f"  <strong>Contagious</strong>: {illness_info['contagious']}\n"
                    result_msg += f"  <strong>Treatment</strong>: {illness_info['treatment']}\n"
                    result_msg += f"  <strong>Notes</strong>: {illness_info['notes']}\n\n"
                    print('\n')
                    printed_illnesses.add(illness_info['illness'])
        if len(sorted_illnesses) > 1:
            result_msg += 'Other possible illnesses:\n'
            for illness, count in sorted_illnesses[1:]:
                if illness not in printed_illnesses:
                    result_msg += f'{illness} ({count} symptoms)\n'
                    for match in matches:
                        for illness_info in map[match]:
                            if illness_info['illness'] == illness and illness_info['illness'] not in printed_illnesses:
                                result_msg += f"  <strong>Symptoms</strong>: {illness_info['symptoms']}\n"
                                result_msg += f"  <strong>Severity</strong>: {illness_info['severity']}\n"
                                result_msg += f"  <strong>Contagious</strong>: {illness_info['contagious']}\n"
                                result_msg += f"  <strong>Treatment</strong>: {illness_info['treatment']}\n"
                                result_msg += f"  <strong>Notes</strong>: {illness_info['notes']}\n\n"
                                print('\n')
                                printed_illnesses.add(illness_info['illness'])
    else:
        result_msg = 'Unable to predict a diagnosis, please try again.'
    return result_msg

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = request.get_json()
        user_input = data['symptoms']
        result_msg = chatbot(user_input)
        return jsonify({'result': result_msg})
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
"""

from flask import Flask, request, jsonify, render_template
import pandas as pd
from collections import defaultdict
import difflib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load dataset
def load_data(file):
    data = pd.read_csv(file)
    return data

# Build symptom-illness map
def build_map(data):
    symptom_map = defaultdict(list)
    known_symptoms = set()
    for _, row in data.iterrows():
        # Ensure Symptoms is a string and clean each symptom
        symptoms = [s.strip().lower() for s in str(row['Symptoms']).split(',')]
        
        illness_info = {
            'illness': row['Illness'],
            'symptoms': row['Symptoms'],  # Original combined symptoms
            'severity': row['Severity'],
            'contagious': row['Contagious'],
            'treatment': row['Treatment'],
            'notes': row['Notes']
        }
        
        for symptom in symptoms:
            cleaned_symptom = symptom.strip()
            if cleaned_symptom:  # Only add non-empty symptoms
                symptom_map[cleaned_symptom].append(illness_info)
                known_symptoms.add(cleaned_symptom)
    
    return symptom_map, known_symptoms

# AI-style symptom extraction using TF-IDF
def extract_symptoms(user_input, data):
    # Get all individual symptoms from dataset
    all_symptoms = set()
    for symptoms in data['Symptoms'].str.lower():
        all_symptoms.update([s.strip() for s in symptoms.split(',')])
    all_symptoms = list(all_symptoms)
    
    # Split user input into potential symptoms (using commas or conjunctions)
    user_symptoms = re.split(r',|\band\b|\bor\b', user_input.lower())
    user_symptoms = [s.strip() for s in user_symptoms if s.strip()]
    
    vectorizer = TfidfVectorizer()
    symptom_vectors = vectorizer.fit_transform(all_symptoms)
    
    matched_symptoms = []
    for user_symptom in user_symptoms:
        user_vector = vectorizer.transform([user_symptom])
        similarities = cosine_similarity(user_vector, symptom_vectors)[0]
        matched_indices = np.where(similarities > 0.1)[0]  # Lower threshold
        matched_symptoms.extend([all_symptoms[i] for i in matched_indices])
    
    return list(set(matched_symptoms))  # Remove duplicates


# approximate matching fallback
def appx_match_symptoms(user_input, symptom_map):
    known_symptoms = list(symptom_map.keys())
    user_input = user_input.lower().strip()
    
    # Split input into potential symptom phrases
    symptom_phrases = re.split(r',|\band\b|\bor\b|with|has|have', user_input)
    symptom_phrases = [phrase.strip() for phrase in symptom_phrases if phrase.strip()]
    
    matches = set()
    
    for phrase in symptom_phrases:
        # 1. Check for direct matches
        for symptom in known_symptoms:
            if (phrase == symptom or 
                phrase in symptom or 
                symptom in phrase or
                any(word in symptom for word in phrase.split())):
                matches.add(symptom)
        
        # 2. Use fuzzy matching
        close_matches = difflib.get_close_matches(
            phrase,
            known_symptoms,
            n=3,
            cutoff=0.5  # Lower threshold
        )
        matches.update(close_matches)
    
    return list(matches)

# Prediction logic
def predict_illnesses(matched_symptoms, symptom_map):
    illness_scores = defaultdict(int)
    for symptom in matched_symptoms:
        for illness_info in symptom_map[symptom]:
            illness_scores[illness_info['illness']] += 1
    return sorted(illness_scores.items(), key=lambda x: x[1], reverse=True)

# Chatbot function
def chatbot(user_input):
    data = load_data('sx.csv')
    symptom_map, _ = build_map(data)
    
    # Step 1: AI-style symptom extraction
    matched_symptoms = extract_symptoms(user_input, data)
    
    # Step 2: appx matching fallback if no matches found
    if not matched_symptoms:
        matched_symptoms = appx_match_symptoms(user_input, symptom_map)  # Pass the full string
    
    if not matched_symptoms:
        return "Couldn't identify specific symptoms. Try phrases like 'headache' or 'stomach pain'."
    
    # Step 3: Predict illnesses
    predicted_illnesses = predict_illnesses(matched_symptoms, symptom_map)
    
    # Format response
    response = "Possible conditions based on your symptoms:\n\n"
    for illness, score in predicted_illnesses[:3]:  # Show top 3
        illness_info = next(info for info in symptom_map[matched_symptoms[0]] 
                      if info['illness'] == illness)
        response += f"**{illness}** (matched {score} symptoms)\n"
        response += f"- Symptoms: {illness_info['symptoms']}\n"
        response += f"- Severity: {illness_info['severity']}\n"
        response += f"- Treatment: {illness_info['treatment']}\n\n"
    
    return response


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = request.get_json()
        user_input = data.get('symptoms', '')
        if not user_input.strip():
            return jsonify({'result': "Please describe your symptoms."})
        result_msg = chatbot(user_input)
        return jsonify({'result': result_msg})
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
