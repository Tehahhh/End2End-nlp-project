import tkinter as tk
from tkinter import Label, Entry, Button, messagebox
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the dataset
df = pd.read_csv("C:\\Users\\WIN10\\Downloads\\emotion_dataset_2.csv")

# Data Cleaning
df = df.dropna(subset=['Clean_Text'])

# Assuming df is your updated training dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['Clean_Text'], df['Emotion'], test_size=0.2, random_state=42
)

# Vectorize the text data using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Build and train the Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_vectorized, y_train)

# Arrays to store user inputs and corresponding emotions
user_inputs = []
user_emotions = []

# GUI setup
window = tk.Tk()
window.title("Emotion Detection Based On Text App")

# Entry widget for user input with a larger width
entry_label = Label(window, text="Please Enter Text:")
entry_label.pack()
entry = Entry(window, width=150)  # Adjusted width to 150
entry.pack()

# Function to predict emotion and store user input
def predict_emotion_and_store():
    user_input = entry.get()
    if user_input:
        cleaned_input = re.sub(r'[^a-zA-Z\s]', '', user_input.lower())
        input_vectorized = vectorizer.transform([cleaned_input])

        # Predict the emotion
        predicted_emotion = nb_model.predict(input_vectorized)[0]

        # Get the probability estimates for each class
        probabilities = nb_model.predict_proba(input_vectorized)[0]
        prediction_percentage = max(probabilities) * 100

        # Store user input and predicted emotion
        user_inputs.append(user_input)
        user_emotions.append(predicted_emotion)

        messagebox.showinfo("Prediction Result", f"Predicted Emotion: {predicted_emotion} "
                                                 f"(Prediction Percentage: {prediction_percentage:.2f}%)")

# Function to display a chart of all emotions in the dataset along with user inputs
# Removed this part as per your request

# Buttons for emotion prediction and display chart
predict_button = Button(window, text="Predict Emotion", command=predict_emotion_and_store)
predict_button.pack()

# Removed the button for displaying the chart

# Run the GUI
window.mainloop()

