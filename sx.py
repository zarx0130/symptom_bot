
from flask import Flask, request, jsonify, render_template
import pandas as pd
from collections import defaultdict
import difflib
import re
import os

app = Flask(__name__)

def load_data(file):
    try:
        data = pd.read_csv(file)
        return data
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        return None

def build_map(data):
    symptom_map = defaultdict(list)
    known_symptoms = set()
    symptom_words = set()
    
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
            for word in re.findall(r'\w+', symptom):
                if len(word) > 2:
                    symptom_words.add(word)
    
    return symptom_map, known_symptoms, symptom_words

def extract_sx(user_input, known_symptoms, symptom_words):
    user_input = user_input.lower()
    extracted_symptoms = set()
    
    for symptom in known_symptoms:
        if re.search(r'\b' + re.escape(symptom) + r'\b', user_input):
            extracted_symptoms.add(symptom)
    
    user_words = re.findall(r'\w+', user_input)
    for word in user_words:
        if word in symptom_words:
            for symptom in known_symptoms:
                if word in symptom and symptom not in extracted_symptoms:
                    extracted_symptoms.add(symptom)
    return list(extracted_symptoms)

def match_sx(user_sx, symptom_map):
    matches = []
    for symptom in user_sx:
        if symptom in symptom_map:
            matches.append(symptom)
        else:
            match = difflib.get_close_matches(symptom, symptom_map.keys(), n=1, cutoff=0.7)
            if match:
                matches.append(match[0])
    return matches

def predict(matches, symptom_map):
    illnesses = defaultdict(int)
    illness_details = {}
    
    for match in matches:
        for illness_info in symptom_map[match]:
            illness = illness_info['illness']
            illnesses[illness] += 1
            if illness not in illness_details:
                illness_details[illness] = illness_info
    return illnesses, illness_details

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = request.get_json()
        user_input = data.get('symptoms', '')
        
        try:
            data = load_data('sx.csv')
            if data is None:
                return jsonify({
                    'primary': None,
                    'secondary': [],
                    'error': 'Failed to load symptom database'
                })

            symptom_map, known_symptoms, symptom_words = build_map(data)
            user_sx = extract_sx(user_input, known_symptoms, symptom_words)
            
            if not user_sx:
                return jsonify({
                    'primary': None,
                    'secondary': [],
                    'error': 'No symptoms detected. Please describe your symptoms.'
                })

            matches = match_sx(user_sx, symptom_map)
            illnesses, illness_details = predict(matches, symptom_map)
            
            if not illnesses:
                return jsonify({
                    'primary': None,
                    'secondary': [],
                    'error': 'No matching illnesses found.'
                })
                
            sorted_illnesses = sorted(illnesses.items(), key=lambda x: x[1], reverse=True)
            
            primary = {
                'name': sorted_illnesses[0][0],
                'count': sorted_illnesses[0][1],
                **illness_details[sorted_illnesses[0][0]]
            }
            
            secondary = [
                {
                    'name': illness,
                    'count': count,
                    **illness_details[illness]
                }
                for illness, count in sorted_illnesses[1:]
            ]
            
            return jsonify({
                'primary': primary,
                'secondary': secondary,
                'error': None
            })
            
        except Exception as e:
            return jsonify({
                'primary': None,
                'secondary': [],
                'error': f'An error occurred: {str(e)}'
            })

if __name__ == '__main__':
    print("Starting symptom checker backend...")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
    if os.path.exists('templates'):
        print(f"Templates found: {os.listdir('templates')}")
    else:
        print("Templates directory missing!")
    app.run(debug=True, port=5000)