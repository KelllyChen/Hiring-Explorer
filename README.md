# Hiring-Explorer
## Motivation
As AI becomes popular, many companies are incorporating it into the early stages of recruitment. They use AI to help HR determine if a candidate meets their requirements. However, this has left job seekers increasingly confused about the actual qualifications companies are looking for. Even worse, the AI models often function like black boxes, and even recruiters don’t fully understand why the system makes certain decisions. As a result, the entire recruitment process can feel discouraging and frustrating.

## Project Overview
This project aims to provide a solution to the above problem. I built a recruiment model and used SHAP to explain the mdoel's decision. To make the explanations more accessible, I incorporated Gemini to interpret the SHAP values in a clear and easy-to-understand way. Users can see an overall explanation of the model, see how each features contribute to the model's predictions, and explore individual rows of data. They can also customize values for each feature to observe how changes affect the outcome. Most importantly, my goal is not to provide a definitive answer about whether the model is biased, but to encourage users to think critically on their own. After exploring the model, users can vote on whether they believe it is biased or not.

## Data
The data was collected from [Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/predicting-hiring-decisions-in-recruitment-data)

## Model
- A RandomForestClassifier was trained based on the 80% of the data, 20% of the data was left as testing set. 
- The accuracy of the model is 93%


## Running Instructions
- Create venv `virtualenv .venv`
- Activate venv `.\.venv\Scripts\activate`
- Install packages `pip install -r requirements.txt`
- Create your own Gemini API key and store in .env file

To recreate the model, run `python model.py`

To run the streamlit app locally, run `streamlit run app.py `

## Live Demo
The app was deployed on streamlit cloud: [Explainable AI for Recruitment Decisions](https://hiring-explorer.streamlit.app/)



