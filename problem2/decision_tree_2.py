#-------------------------------------------------------------------------
# AUTHOR: Michael Castillo
# FILENAME: decision_tree_2.py
# SPECIFICATION: Train, test and output the performance of the 3 models
# FOR: CS 4210- Assignment #2
# TIME SPENT: 30 minutes
#-----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas.
# You have to work here only with standard dictionaries, lists, and arrays.

# Importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

# Mapping categorical values to numerical values
categoryMap = {
    "Young": 1, "Prepresbyopic": 2, "Presbyopic": 3,
    "Myope": 1, "Hypermetrope": 2,
    "Yes": 1, "No": 2,
    "Reduced": 1, "Normal": 2
}

# Read test data
dbTest = []
with open('contact_lens_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # Skipping the header
            dbTest.append(row)

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    # Reading the training data from a CSV file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0:  # Skipping the header
                dbTraining.append(row)

    # Transform the original categorical training features to numbers and add to the 4D array X.
    for row in dbTraining:
        X.append([categoryMap[row[0]], categoryMap[row[1]], categoryMap[row[2]], categoryMap[row[3]]])

    # Transform the original categorical training classes to numbers and add to the vector Y.
    for row in dbTraining:
        Y.append(categoryMap[row[4]])

    total_accuracy = 0

    # Loop training and testing tasks 10 times
    for i in range(10):

        # Fitting the decision tree to the data, setting max_depth=5
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
        clf = clf.fit(X, Y)

        correct_predictions = 0
        total_predictions = 0

        for data in dbTest:
            # Transform test features using the same numerical mapping
            test_instance = [categoryMap[data[0]], categoryMap[data[1]], categoryMap[data[2]], categoryMap[data[3]]]
            class_predicted = clf.predict([test_instance])[0]

            # Compare prediction with true label
            true_label = categoryMap[data[4]]
            if class_predicted == true_label:
                correct_predictions += 1

            total_predictions += 1

        # Calculate accuracy for this run
        accuracy = correct_predictions / total_predictions
        total_accuracy += accuracy

    # Find the average accuracy for the 10 runs
    avg_accuracy = total_accuracy / 10

    # Print the average accuracy of this model
    print(f'Final accuracy when training on {ds}: {avg_accuracy:.2f}')




