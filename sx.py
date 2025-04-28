
from flask import Flask, request, jsonify, render_template
import pandas as pd
from collections import defaultdict
import difflib
import re

app = Flask(__name__)

# load dataset
def load_data(file):
    data = pd.read_csv(file)  # sx.csv
    return data

# symptom-illness map 
def build_map(data):
    symptom_map = defaultdict(list)
    known_symptoms = set() 
    symptom_words = set()  # All individual words in symptoms
    
    for _, row in data.iterrows():
        symptoms = [s.strip().lower() for s in row['Symptoms'].split(',')]
        illness_info = {
            'illness': row['Illness'],
            'symptoms': row["Symptoms"],
            'severity': row['Severity'],
            'contagious': row['Contagious'],
            'treatment': row['Treatment'],
            'notes': row['Notes']
        }
        
        for symptom in symptoms:
            symptom_map[symptom].append(illness_info)
            known_symptoms.add(symptom)
            # Simple word splitting without NLTK
            for word in re.findall(r'\w+', symptom):
                if len(word) > 2:  # Ignore very short words
                    symptom_words.add(word)
    
    return symptom_map, known_symptoms, symptom_words

# extract symptoms from user input
def extract_sx(user_input, known_symptoms, symptom_words):
    user_input = user_input.lower()
    extracted_symptoms = set()
    
    # First check for exact multi-word symptom matches
    for symptom in known_symptoms:
        if re.search(r'\b' + re.escape(symptom) + r'\b', user_input):
            extracted_symptoms.add(symptom)
    
    # Then check for individual words that appear in known symptoms
    user_words = re.findall(r'\w+', user_input)
    for word in user_words:
        if word in symptom_words:
            # Find all symptoms that contain this word
            for symptom in known_symptoms:
                if word in symptom and symptom not in extracted_symptoms:
                    extracted_symptoms.add(symptom)
    print("extracted_symptoms:", extracted_symptoms)
    return list(extracted_symptoms)

# match user-provided symptoms to known symptoms (with spelling tolerance)
def match_sx(user_sx, symptom_map):
    matches = []
    for symptom in user_sx:
        # First try exact match
        if symptom in symptom_map:
            matches.append(symptom)
        else:
            # Then try fuzzy match
            match = difflib.get_close_matches(symptom, symptom_map.keys(), n=1, cutoff=0.7)
            if match:
                matches.append(match[0])
    return matches

# predict illness based on matched symptoms
def predict(matches, symptom_map):
    illnesses = defaultdict(int)
    illness_details = {}  # Store all illness info
    
    for match in matches:
        for illness_info in symptom_map[match]:
            illness = illness_info['illness']
            illnesses[illness] += 1
            # Store details if not already present
            if illness not in illness_details:
                illness_details[illness] = illness_info
    
    return illnesses, illness_details  # Return both counts and details

def chatbot(user_input):
    try:
        data = load_data('sx.csv')
        symptom_map, known_symptoms, symptom_words = build_map(data)
        user_sx = extract_sx(user_input, known_symptoms, symptom_words)
        
        if not user_sx:
            return {
                'primary': 'No symptoms detected. Please try again.',
                'secondary': []
            }

        matches = match_sx(user_sx, symptom_map)
        illnesses, illness_details = predict(matches, symptom_map)
        
        if not illnesses:
            return {
                'primary': 'No matching illnesses found.',
                'secondary': []
            }
            
        # Sort by match count (descending)
        sorted_illnesses = sorted(illnesses.items(), key=lambda x: x[1], reverse=True)
        
        # Format primary diagnosis
        primary_illness, primary_count = sorted_illnesses[0]
        primary_info = illness_details[primary_illness]
        primary_response = format_illness_response(primary_illness, primary_count, primary_info)
        
        # Format secondary diagnoses
        secondary_responses = []
        for illness, count in sorted_illnesses[1:]:
            info = illness_details[illness]
            secondary_responses.append(format_illness_response(illness, count, info))
        
        return {
            'primary': primary_response,
            'secondary': secondary_responses
        }
        
    except Exception as e:
        return {
            'primary': f"Error: {str(e)}",
            'secondary': []
        }

def format_illness_response(illness, count, info):
    return (
        f"{illness} (matching {count} symptoms)\n"
        f"Symptoms: {info['symptoms']}\n"
        f"Severity: {info['severity']}\n"
        f"Contagious: {info['contagious']}\n"
        f"Treatment: {info['treatment']}\n"
        f"Notes: {info['notes']}"
    )

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = request.get_json()
        user_input = data['symptoms']
        result = chatbot(user_input)
        return jsonify(result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)