from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import math

# Create your views here.


def index(request):
    result = ''
    accuracy = ''
    try:
        if request.method == "POST":

            # Sample dataset
            data = pd.read_csv('E:/Grace data/ML Project/24_heartattackprediction/ht.csv')
            # data = pd.read_csv('E:/Grace data/ML Project/HeartAttakPredictor/ht.csv')
            #E:\Grace data\ML Project\HeartAttakPredictor
            #data = pd.read_csv('E:/ht.csv')
            # Define features (X) and target variable (y)
            X = data.drop('target', axis=1)
            y = data['target']

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            # Standardize features (recommended for SVM)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Create and train the SVM classifier
            svm_classifier = SVC(kernel='linear', C=1)
            svm_classifier.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = svm_classifier.predict(X_test)

            # Calculate and display the accuracy of the model
            acc = accuracy_score(y_test, y_pred)
            accuracy = math.ceil(acc*100)
            print("Accuracy:", accuracy, "%")

            # Allow the user to input new data for prediction
            # print("\nEnter new data for prediction:")
            new_data = []
            age = eval(request.POST.get('age'))
            new_data.append(age)
            sex = eval(request.POST.get('sex'))
            new_data.append(sex)
            cp = eval(request.POST.get('cp'))
            new_data.append(cp)
            tbps = eval(request.POST.get('trestbps'))
            new_data.append(tbps)
            chol = eval(request.POST.get('chol'))
            new_data.append(chol)
            fbs = eval(request.POST.get('fbs'))
            new_data.append(fbs)
            restsge = eval(request.POST.get('restecg'))
            new_data.append(restsge)
            thala = eval(request.POST.get('thalach'))
            new_data.append(thala)
            exang = eval(request.POST.get('exang'))
            new_data.append(exang)
            odl = eval(request.POST.get('oldpeak'))
            new_data.append(odl)
            slope = eval(request.POST.get('slope'))
            new_data.append(slope)
            ca = eval(request.POST.get('cp'))
            new_data.append(ca)
            thal = eval(request.POST.get('thal'))
            new_data.append(thal)

            # Standardize the user's input data
            new_data = scaler.transform([new_data])

            # Predict the output based on user input
            user_prediction = svm_classifier.predict(new_data)

            if user_prediction[0] == 0:
                result = "Feel Free, You Don't Have Heart Attack "+" \u2764\uFE0F"+'\U0001F604'
            else:
                result = "Yes, You Have Heart Attack "+'\U0001F494'+'\U0001F622'

            print(f"Prediction: {result}")

    except:
        result = "Please Provide Valid Inputs " + '\U0001F620' + '\U0001F620'

    return render(request, "index.html", {'result': result})
