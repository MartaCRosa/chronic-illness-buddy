# Chronic Illness Buddy

Chronic Illness Buddy is a demo app designed to assist users in managing their health, for now, only for the illness of diabetes. It provides three main features:  
- A **Glucose Level Predictor** that estimates potential hypoglicemc or hyperglicemic states based on user inputs.  
- A **Medication Manager** that notifies the user of medication timing.
- An **AI Chatbot** powered by OpenAI's API to answer health-related questions.

## Features

### Glucose Level Predictor
The glucose level predictor uses user-provided details to predict and classify the state of the glucole level. The required inputs include:  
- Age
- Gender  
- Weight
- Height   
- Heart Rate  
- Hours from last eaten meal  

It leverages a Fully Connected Multi-Layer Neural Network model that has been trained with a glucole level dataset found on Kaggle (www.kaggle.com/datasets/fatimaafzaal/glucose-level-estimation), that then is used to predict one of three outcomes: Hypoglicemia, Normal Glucose Level or Hyperglicemia.

### Medication Manager
The medication manager let's the user insert all of the medication they take and the associated details:
- Name
- Type (Oral, Transdermal, Nasal, Other)
- Time of administration

The notification part of this module is yet to be implemented, but the timings would connect to the device's push notifications.

### AI Chatbot
The AI chatbot connects to OpenAI's API to provide  conversational responses or advice to health-related questions.  
- Ask general health queries.  
- Learn about managing diabetes.  
- Get tips on improving daily health routines.

## Technologies
- Python
- Flask 
- Machine Learning for the model predictor
- OpenAI API for the chatbot
- Docker for containerization

## Installation

### Prerequisites
Ensure you have the following:  
- Python 3.11  
- Docker (for containerized deployment)  
- An OpenAI API key defined in the environment variables as OPENAI_API_KEY

### Steps to Run
Follow these steps to run the app:
1. Install requirements by running 'pip install -r requirements.txt'
2. Run the file run.py, 'python run.py'
3. The app will be accessible at 'http://localhost:5000'
