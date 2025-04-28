
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
    for match in matches:
        for illness_info in symptom_map[match]:
            illness = illness_info['illness']
            illnesses[illness] += 1
    return illnesses

# chatbot function
def chatbot(user_input):
    try:
        # load dataset
        data = load_data('sx.csv')
        # build symptom-illness map
        symptom_map, known_symptoms, symptom_words = build_map(data)
        # extract symptoms from user input
        user_sx = extract_sx(user_input, known_symptoms, symptom_words)
        
        if not user_sx:
            return 'No symptoms detected. Please describe how you\'re feeling, like "headache and fever".'

        # match symptoms
        matches = match_sx(user_sx, symptom_map)
        # predict illness
        illnesses = predict(matches, symptom_map)
        
        if not illnesses:
            return 'No matching illnesses found. Please try describing your symptoms differently.'
            
        # sort illnesses by match count
        sorted_illnesses = sorted(illnesses.items(), key=lambda x: x[1], reverse=True)
        
        # prepare response
        result_msg = 'Possible conditions based on your symptoms:\n\n'
        printed_illnesses = set()
        
        for illness, count in sorted_illnesses:
            if illness in printed_illnesses:
                continue
                
            result_msg += f'<strong>{illness}</strong> (matching {count} symptoms)\n'
            
            # Find the first matching symptom info for this illness
            for match in matches:
                for info in symptom_map[match]:
                    if info['illness'] == illness:
                        result_msg += f"  Symptoms: {info['symptoms']}\n"
                        result_msg += f"  Severity: {info['severity']}\n"
                        result_msg += f"  Contagious: {info['contagious']}\n"
                        result_msg += f"  Treatment: {info['treatment']}\n"
                        result_msg += f"  Notes: {info['notes']}\n\n"
                        printed_illnesses.add(illness)
                        break
                if illness in printed_illnesses:
                    break
                    
        return result_msg
    
    except Exception as e:
        return f"An error occurred: {str(e)}. Please try again later."

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