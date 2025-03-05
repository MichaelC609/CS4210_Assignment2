#-------------------------------------------------------------------------
# AUTHOR: Michael Castillo
# FILENAME: naive_bayes.py
# SPECIFICATION: Naive Bayes classifier for weather data classification
# FOR: CS 4210- Assignment #2
# TIME SPENT: 30 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

# Function to map categorical features to numerical values
def encode_feature(value, mapping):
    return mapping[value]

# Define mappings for categorical attributes
outlook_mapping = {"Sunny": 1, "Overcast": 2, "Rain": 3}
temperature_mapping = {"Hot": 1, "Mild": 2, "Cool": 3}
humidity_mapping = {"High": 1, "Normal": 2}
wind_mapping = {"Weak": 1, "Strong": 2}
playtennis_mapping = {"Yes": 1, "No": 2}

#Reading the training data in a csv file
X = []
Y = []

with open("weather_training.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        X.append([
            encode_feature(row[1], outlook_mapping),
            encode_feature(row[2], temperature_mapping),
            encode_feature(row[3], humidity_mapping),
            encode_feature(row[4], wind_mapping)
        ])
        Y.append(encode_feature(row[5], playtennis_mapping))

#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
test_data = []
with open("weather_test.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        test_data.append(row)

#Printing the header of the solution
print("Day Outlook Temperature Humidity Wind PlayTennis Confidence")

#Use your test samples to make probabilistic predictions
for row in test_data:
    features = [
        encode_feature(row[1], outlook_mapping),
        encode_feature(row[2], temperature_mapping),
        encode_feature(row[3], humidity_mapping),
        encode_feature(row[4], wind_mapping)
    ]
    probabilities = clf.predict_proba([features])[0]
    prediction = clf.predict([features])[0]
    confidence = max(probabilities)
    
    if confidence >= 0.75:
        predicted_class = "Yes" if prediction == 1 else "No"
        print(f"{row[0]} {row[1]} {row[2]} {row[3]} {row[4]} {predicted_class} {confidence:.2f}")
