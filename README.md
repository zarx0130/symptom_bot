# symptom_bot
Flask-based web application that helps users identify potential illnesses based on experienced symptoms using natural language processing and symptom matching algorithms

# Features
Natural Language Processing - understands symptom descriptions in plain English
Symptom Matching - uses fuzzy matching to handle variations in descriptions
Illness Prediction - ranks potential illnesses by symptom match count
Detailed Information - provides severity, contagiousness, treatment options, and notes for each illness
Responsive Web Interface - user-friendly frontend for symptom input

# Technology
Backend - Flask (Python)
Frontend - HTML, CSS, JavaScript
Data Processing - Pandas
NLP - regular expressions, difflib for fuzzy match
Data Storage - CSV for illness-symptom database (limited for demo)

# Usage
install dependencies - pip install [library/requirements]
ex: pip install flask pandas

to run - python app.py (or in some cases python3 app.py)
open browser and navigate to http://localhost:5000 (/ping for health checks)

1. Describe symptoms in text box (ex: the left side of my head aches)
2. Submit to get predictions
3. View results showing: primary prediction (most likely match), secondary predictions, and details for each

# Database
sx.csv file should include the following columns:
Column	Description	Example
Illness	Name of the illness	"Cold"
Symptoms	Comma-separated list of symptoms	"Sneezing, sore throat, runny nose"
Severity	Illness severity level	"Mild", "Moderate", "Severe"
Contagious	Whether the illness is contagious	"Yes", "No", "Sometimes"
Treatment	Recommended treatment options	"Rest, fluids, decongestants"
Notes	Additional information	"Caused by viral infections"

*to add new rows following same format: ensure symptoms are comma separated and consistently formatted, application will automatically reload the data on startup

# Disclaimer
This symptom checker is for informational and educational purposes only, it is not a substitute for professional medical advice, diagnosis, or treatment. Seek the advice of a physician or other qualified health provider with any questions you may have regarding a medical condition. 
