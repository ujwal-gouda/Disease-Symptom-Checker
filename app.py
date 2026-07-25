from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

app = Flask(__name__)

neuralnetModel = load_model('disease_symptom_checker_NN_model.keras')

# Load disease label mapping from the dataset so numeric predictions can be decoded
label_df = pd.read_csv('Final_Augmented_dataset_Diseases_and_Symptoms.csv')
le = LabelEncoder()
le.fit(label_df['diseases'])

# Get symptoms from dataset feature columns
all_symptoms = list(label_df.drop(columns=['diseases']).columns)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        selected_symptoms = request.form.getlist("symptoms")

        input_data = pd.DataFrame(
            0,
            index=[0],
            columns=all_symptoms
        )

        for symptom in selected_symptoms:
            if symptom in input_data.columns:
                input_data[symptom] = 1

        raw_probs = neuralnetModel.predict(input_data)
        raw_prediction = np.argmax(raw_probs, axis=1)[0]
        prediction = le.inverse_transform([raw_prediction])[0]

    return render_template(
        "index.html",
        symptoms=all_symptoms,
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)