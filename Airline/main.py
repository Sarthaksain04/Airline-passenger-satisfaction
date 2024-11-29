from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('model.lb')

# Load your CSV data
dataset = pd.read_csv('test.csv')

# Ensure columns have no extra spaces
dataset.columns = dataset.columns.str.strip()

# Define the threshold for satisfaction
satisfaction_threshold = 3  # Modify as needed

# Function to classify satisfaction based on rating
def classify_satisfaction(rating):
    return "Satisfied" if rating >= satisfaction_threshold else "Unsatisfied"

# Apply satisfaction classification to relevant columns
dataset['WifiServiceSatisfaction'] = dataset['Inflight wifi service'].apply(classify_satisfaction)
dataset['SeatComfortSatisfaction'] = dataset['Seat comfort'].apply(classify_satisfaction)
dataset['FoodAndDrinkSatisfaction'] = dataset['Food and drink'].apply(classify_satisfaction)
dataset['EntertainmentSatisfaction'] = dataset['Inflight entertainment'].apply(classify_satisfaction)

@app.route('/')
def index():
    return render_template('Feedbackform.html')

@app.route('/submit', methods=['POST'])
def submit_form():
    # Get form data
    email = request.form.get('email')
    gender = request.form.get('gender')
    age = request.form.get('age')
    travel_type = request.form.get('type-of-travel')
    class_type = request.form.get('class')
    wifi_service = request.form.get('wifi-service')
    seat_comfort = request.form.get('seat-comfort')
    food_drink = request.form.get('food-drink')
    entertainment = request.form.get('entertainment')
    ground_service = request.form.get('ground-service')
    cabin_crew_service = request.form.get('cabin-crew-service')
    arrival_delay = request.form.get('arrival-delay')

    # Convert numeric fields to integers, handling possible missing data
    numeric_fields = [age, wifi_service, seat_comfort, food_drink, entertainment, ground_service, cabin_crew_service, arrival_delay]
    numeric_values = [int(field) if field else 0 for field in numeric_fields]

    # Prepare the input for prediction
    latest_data_values = numeric_values  # Replace with correct column ordering for model input
    prediction = model.predict([latest_data_values])[0]
    labels = {1: 'SATISFIED', 0: 'DISSATISFIED'}
    prediction_result = labels[prediction]

    # Filter the dataset based on the form inputs
    filtered_data = dataset[(
        dataset['Gender'] == gender) & 
        (dataset['Customer Type'] == class_type) & 
        (dataset['Age'] == int(age)) & 
        (dataset['Type of Travel'] == travel_type)
    ]

    # Create a list of columns to include in the results
    result_columns = [
        'Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance',
        'WifiServiceSatisfaction', 'SeatComfortSatisfaction', 'FoodAndDrinkSatisfaction',
        'EntertainmentSatisfaction', 'Leg room service', 'Inflight service', 'Arrival Delay in Minutes'
    ]

    # Get the filtered data for the relevant columns
    result_data = filtered_data[result_columns]

    # Prepare data for rendering
    table_headers = result_data.columns.tolist()
    table_rows = result_data.values.tolist()

    # Render the results page
    return render_template(
        'results.html',
        email=email,
        prediction=prediction_result,
        table_headers=table_headers,
        table_rows=table_rows
    )

if __name__ == '__main__':
    app.run(debug=True)
