from flask import Flask, render_template, request

import joblib
import numpy as np

app = Flask(__name__)

# LOAD MODEL
classifier_model = joblib.load('models/classifier_model.pkl')
scaler = joblib.load('models/scaler.pkl')
le_gender = joblib.load('models/le_gender.pkl')
le_workout = joblib.load('models/le_workout.pkl')

# Halaman utama
@app.route('/')
def home():
    return render_template('index.html')

# Halaman prediksi
@app.route('/predict', methods=['POST'])
def predict():
    # INPUT USER
    age = float(request.form['age'])
    gender = request.form['gender']
    weight = float(request.form['weight'])
    height = float(request.form['height'])
    max_bpm = float(request.form['max_bpm'])
    avg_bpm = float(request.form['avg_bpm'])
    resting_bpm = float(request.form['resting_bpm'])
    session_duration = float(
        request.form['session_duration']
    )
    calories_burned = float(
        request.form['calories_burned']
    )
    fat_percentage = float(
        request.form['fat_percentage']
    )
    water_intake = float(
        request.form['water_intake']
    )
    workout_frequency = float(
        request.form['workout_frequency']
    )
    bmi = float(request.form['bmi'])
    workout_type = request.form['workout_type']

    # ENCODING
    gender_enc = le_gender.transform([gender])[0]

    workout_enc = le_workout.transform(
        [workout_type]
    )[0]

    # URUTAN HARUS SAMA DENGAN TRAINING
    data = [[
        age,
        weight,
        height,
        max_bpm,
        avg_bpm,
        resting_bpm,
        session_duration,
        calories_burned,
        fat_percentage,
        water_intake,
        workout_frequency,
        bmi,
        gender_enc,
        workout_enc
    ]]
    # SCALING
    data_scaled = scaler.transform(data)
    # PREDIKSI
    prediction = classifier_model.predict(
        data_scaled
    )[0]
    # LABEL CLUSTER
    cluster_labels = {
        0: "Elite Athlete",
        1: "Advanced Trainer",
        2: "Moderate Fitness",
        3: "Casual Beginner"
    }
    result = cluster_labels.get(
        prediction,
        "Unknown"
    )
    # INSIGHT & RECOMMENDATION
    insights = {

        "Elite Athlete": {

            "insight": [
                "✔ High workout consistency",
                "✔ Excellent cardiovascular endurance",
                "✔ Low body fat percentage",
                "✔ High calorie burn efficiency"
            ],

            "recommendation":
            "Maintain progressive overload training and prioritize recovery."
        },

        "Advanced Trainer": {

            "insight": [
                "✔ Strong fitness discipline",
                "✔ Good training intensity",
                "✔ Balanced physical condition",
                "✔ Stable workout frequency"
            ],

            "recommendation":
            "Increase workout variation and improve stamina endurance."
        },

        "Moderate Fitness": {

            "insight": [
                "✔ Moderate physical activity",
                "✔ Balanced exercise habits",
                "✔ Average calorie burn",
                "✔ Developing fitness performance"
            ],

            "recommendation":
            "Increase workout consistency and improve nutrition habits."
        },

        "Casual Beginner": {

            "insight": [
                "✔ Beginner fitness level",
                "✔ Lower workout intensity",
                "✔ Early fitness development",
                "✔ Requires structured training"
            ],

            "recommendation":
            "Start with regular light exercise and improve workout consistency."
        }
    }
    selected_insight = insights[result]
    return render_template(
        'result.html',
        result=result,
        insight_list=selected_insight["insight"],
        recommendation=selected_insight["recommendation"],
        bmi=bmi,
        calories=calories_burned,
        workout_freq=workout_frequency,
        fat=fat_percentage
    )

if __name__ == '__main__':
    app.run(debug=True)
