from LLM.chatbot import generate_response
from flask import Blueprint, render_template, request, redirect, url_for
from Predictor.glucose_prediction import classify_glucose
import os
import json

main = Blueprint('main', __name__)

# File to store conversation
CHAT_FILE = "conversation.json"
MEDICATION_FILE = "medication.json"

# Load the conversation from the file
def load_json(file_name):
    if os.path.exists(file_name):
        with open(file_name, "r") as file:
            return json.load(file)
    return []

# Save the conversation to the file
def save_json(text, file_name):
    with open(file_name, "w") as file:
        json.dump(text, file)


# Route to home page
@main.route('/')
def home():
    return render_template('index.html')


# Route to chatbot page
@main.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        if user_input:
            # Load existing conversation
            conversation = load_json(CHAT_FILE)
            # Add user input and bot response to the conversation
            conversation.append({"sender": "You", "message": user_input})
            response = generate_response(user_input)
            conversation.append({"sender": "Chatbot Buddy", "message": response})
            # Save the updated conversation
            save_json(conversation, CHAT_FILE)
        # Redirect to the GET version of the page to get input
        return redirect(url_for('main.chatbot'))

    # Handle GET request
    conversation = load_json(CHAT_FILE)
    return render_template('chatbot.html', conversation=conversation)

# Route to cleared chatbot page
@main.route('/clear_chat')
def clear_chat():
    print("clearing conversation")
    save_json([], CHAT_FILE)  # Clear the conversation
    return redirect(url_for('main.chatbot'))


# Route to details page
@main.route('/details', methods=['GET', 'POST'])
def details():
    if request.method == 'POST':
        # Get data from form 
        age = float(request.form['age'])
        gender = request.form['gender']
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        heart_rate = float(request.form['heart_rate'])

        # Get hours and minutes from dropdown
        last_eaten_hours = float(request.form['last_eaten_hours'])
        last_eaten_minutes = float(request.form['last_eaten_minutes'])

        # Calculate total last eaten time in hours
        last_eaten = last_eaten_hours + last_eaten_minutes

        # Validation logic (server-side)
        errors = []      
        
        if not (15 <= age <= 110):
            errors.append("Age must be between 15 and 120.")  # Check for valid age
             
        if not (20 <= weight <= 200):
            errors.append("Weight must be between 20kg and 200kg.")  # Check for valid weight
              
        if not (100 <= height <= 250):
            errors.append("Height must be between 1m and 2.5m.")  # Check for valid height
           
        if not (10 <= heart_rate <= 200):
            errors.append("Heart rate must be between 10bpm and 200bpm.")  # Check for valid heart rate
        
        # Handle 'last eaten' time logic
        if last_eaten > 8:
            last_eaten = -1  # Default to -1 if more than 8 hours
      
        if not (0 <= last_eaten <= 8 or last_eaten == -1):
            errors.append("Last eaten time must be between 0 and 8 hours. Fasting is -1.") # Check if last_eaten is in the valid range (If above 8 default to -1)

        if errors:
            # If there are errors, render the form again with the error messages
            return render_template('details.html', errors=errors)
    
        # Render result
        user_data = {
            "age": age,
            "gender": gender,
            "weight": weight,
            "height": height,
            "heart_rate": heart_rate,
            "last_eaten": last_eaten
        }
        diabetic = "Y"  # Always set diabetic status to "Yes"

        # Call prediction function
        try:
            glucose_class = classify_glucose(user_data['age'], user_data['gender'], user_data['weight'], user_data['heart_rate'], user_data['height']*0.032808, user_data['last_eaten'], diabetic)
        except Exception as e:
            errors.append(f"Error during prediction: {str(e)}")
            return render_template('details.html', errors=errors)

        return render_template('details.html', user_data=user_data, glucose_class=glucose_class)
    
    # If it's a GET request, just render the form (client hasn't inserted data)
    return render_template('details.html')


# Route to reminder page
@main.route('/reminder', methods=['GET', 'POST'])
def reminder():
    if request.method == 'POST':
        # Get name and type from form
        medication = request.form['medication']
        medication_administration = request.form['medication-administration']

        # Get hours and minutes from dropdown
        hours = float(request.form['hours'])
        minutes = float(request.form['minutes'])

        # Validation logic (server-side)
        errors = []    
        if errors:
            # If there are errors, render the form again with the error messages
            return render_template('reminder.html', errors=errors)
    
        # Render result
        medication_data = {
            "medication": medication,
            "medication_administration": medication_administration,
            "hours": hours,
            "minutes": minutes
        }
        
        temp = load_json(MEDICATION_FILE)
        if temp:
            medication_data_new=[]
            medication_data_new.extend([medication_data, temp])
            save_json(medication_data_new, MEDICATION_FILE)
        else:
            save_json(medication_data, MEDICATION_FILE)  

    # If it's a GET request, just render the form (client hasn't inserted data)
    return render_template('reminder.html')
