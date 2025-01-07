from LLM.chatbot import generate_response
from flask import Blueprint, render_template, request, redirect, url_for
import os
import json
from Predictor.glucose_prediction import classify_glucose

main = Blueprint('main', __name__)

# File to store conversation
CHAT_FILE = "conversation.json"

def load_conversation():
    """Load the conversation from the file."""
    if os.path.exists(CHAT_FILE):
        with open(CHAT_FILE, "r") as file:
            return json.load(file)
    return []

def save_conversation(conversation):
    """Save the conversation to the file."""
    with open(CHAT_FILE, "w") as file:
        json.dump(conversation, file)

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        if user_input:
            # Load existing conversation
            conversation = load_conversation()
            # Add user input and bot response to the conversation
            conversation.append({"sender": "User", "message": user_input})
            response = generate_response(user_input)
            conversation.append({"sender": "Chatbot buddy", "message": response})
            # Save the updated conversation
            save_conversation(conversation)
        # Redirect to the GET version of the page
        return redirect(url_for('main.chatbot'))

    # Handle GET request
    conversation = load_conversation()
    return render_template('chatbot.html', conversation=conversation)

@main.route('/details', methods=['GET', 'POST'])
def details():
    if request.method == 'POST':
        # Get form data
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
        
        # Check for valid age
        if not (1 <= age <= 110):
            errors.append("Age must be between 1 and 120.")
        
        # Check for valid weight
        if not (20 <= weight <= 200):
            errors.append("Weight must be between 20kg and 200kg.")
        
        # Check for valid height
        if not (100 <= height <= 250):
            errors.append("Height must be between 1m and 2.5m.")
        
        # Check for valid heart rate
        if not (20 <= heart_rate <= 200):
            errors.append("Heart rate must be between 20bpm and 200bpm.")
        
        # Handle 'last eaten' time logic
        if last_eaten > 8:
            last_eaten = -1  # Default to -1 if more than 8 hours

        # Check if last_eaten is in the valid range
        if not (0 <= last_eaten <= 8 or last_eaten == -1):
            errors.append("Last eaten time must be between 0 and 8 hours. If above 8, default to -1.")

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
        
        # Always set diabetic status to "Yes"
        diabetic = "Y"

        # Call prediction function
        try:
            glucose_class = classify_glucose(user_data['age'], user_data['gender'], user_data['weight'], user_data['heart_rate'], user_data['height']*0.032808, user_data['last_eaten'], diabetic)
        except Exception as e:
            errors.append(f"Error during prediction: {str(e)}")
            return render_template('details.html', errors=errors)

        return render_template('details.html', user_data=user_data, glucose_class=glucose_class)
    
    # If it's a GET request, just render the form
    return render_template('details.html')


@main.route('/clear_chat')
def clear_chat():
    print("clearing conversation")
    save_conversation([])  # Clear the conversation
    return redirect(url_for('main.chatbot'))



