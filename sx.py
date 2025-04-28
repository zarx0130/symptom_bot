
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