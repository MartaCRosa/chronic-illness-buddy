# Function to classify glucose levels based on eating timing
def classify_glucose_by_eating(last_eaten, glucose_level):
    if last_eaten == -1:  # Fasting
        if glucose_level > 130:
            return "Hyperglycemia"
        elif 80 <= glucose_level <= 130:
            return "Normal"
        else:
            return "Hypoglycemia"
    elif last_eaten >= 0:  # Between 0 and 2 hours after eating
        if glucose_level > 180:
            return "Hyperglycemia"
        elif 80 <= glucose_level <= 180:
            return "Normal"
        else:
            return "Hypoglycemia"
    else:
        return "Unknown"  # Handle unexpected cases
    
# Optional: Predict on a new example