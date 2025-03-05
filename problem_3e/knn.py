#-------------------------------------------------------------------------
# AUTHOR: Michael Castillo
# FILENAME: knn.py
# SPECIFICATION: comupte the LOO-CV error rate for a 1NN classifier
# FOR: CS 4210- Assignment #2
# TIME SPENT: 20 minutes
#-----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas.
# You have to work here only with standard vectors and arrays.

# Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

# Reading the data in a csv file
with open('email_classification.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skipping the header
    for row in reader:
        db.append(row)

# Initialize error count
error_count = 0

total_instances = len(db)

# Loop through each instance as a test set
for i in range(total_instances):
    # Add the training features to the 20D array X, removing the test instance
    X = [list(map(float, row[:-1])) for j, row in enumerate(db) if j != i]
    
    # Transform the original training classes to numbers and add them to Y
    Y = [1 if row[-1] == "spam" else 0 for j, row in enumerate(db) if j != i]
    
    # Store the test sample of this iteration
    test_sample = list(map(float, db[i][:-1]))
    true_label = 1 if db[i][-1] == "spam" else 0

    # Fitting the 1NN classifier to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf.fit(X, Y)

    # Use the test sample in this iteration to make the class prediction
    class_predicted = clf.predict([test_sample])[0]

    # Compare the prediction with the true label
    if class_predicted != true_label:
        error_count += 1

# Compute and print the error rate
error_rate = error_count / total_instances
print("LOO-CV Error Rate:", error_rate)






