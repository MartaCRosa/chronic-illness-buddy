def classify_glucose(last_eaten, glucose_level):
    if last_eaten == -1:  # Fasting
        if glucose_level > 130:
            return "Hyperglycemia"
        elif 80 <= glucose_level <= 130:
            return "Normal"
        else:
            return "Hypoglycemia"
    elif 0 < last_eaten < 4:  # Between 0 and 2 hours after eating
        if glucose_level > 180:
            return "Hyperglycemia"
        elif 90 <= glucose_level <= 180:
            return "Normal"
        else:
            return "Hypoglycemia"

    elif last_eaten >= 4:  # Between 0 and 2 hours after eating
        if glucose_level > 150:
            return "Hyperglycemia"
        elif 90 <= glucose_level <= 150:
            return "Normal"
        else:
            return "Hypoglycemia"
    else:
        return "Unknown"