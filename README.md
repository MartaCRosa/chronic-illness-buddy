how often should i train y model?
permutation importance
add "inform emergency contacts when hypo or hyper"

# Chronic Illness Buddy

Chronic Illness Buddy is a demo app designed to assist users in managing their health, for now, only for the illness of diabetes. It provides two main features:  
- A **Glucose Level Predictor** that estimates potential hypoglicemc or hyperglicemic states based on user inputs.  
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

It leverages a Fully Connected Feedforward Neural Network model that has been trained with a glucole level dataset found on Kaggle (www.kaggle.com/datasets/fatimaafzaal/glucose-level-estimation), that then is used to predict one of three outcomes: Hypoglicemia, Normal Glucose Level or Hyperglicemia.

### AI Chatbot
The AI chatbot connects to OpenAI's API to provide reliable, conversational responses to health-related questions.  
- Ask general health queries.  
- Learn about managing diabetes.  
- Get tips on improving daily health routines.

## Technologies
- Python
- Flask for the backend
- Machine Learning (Scikit-learn and TensorFlo) for the model predictor
- OpenAI API for the chatbot
- Docker for containerization

## Installation

### Prerequisites
Ensure you have the following installed:  
- Python 3.8+  
- Docker (for containerized deployment)  
- An OpenAI API key ([sign up for one here](https://platform.openai.com/signup/))  

### Steps to Run
1. Install requirements on requirements.txt
2. Run the file run.py
3. The app will be accessible at 'http://localhost:5000'
