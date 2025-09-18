# Predict-Titanic-Survival-with-Logistic-Regression
# What I Built?
This project is a machine learning model that predicts survival outcomes for passengers on the Titanic based on various features such as age, sex, class, and more. The model utilizes a logistic regression algorithm to classify passengers as survivors or non-survivors.

# Why I Built It?
I built this project to demonstrate my understanding of machine learning concepts, specifically classification tasks, and to practice working with datasets. This project showcases how to:

- Load and preprocess data
- Build and train a logistic regression model
- Evaluate the model's performance using accuracy and classification reports
- Visualize the model's performance using ROC curves

# How I Built It?
1. Data Loading: I loaded the Titanic dataset using pandas and removed unnecessary columns such as Name, Ticket, and Cabin.
2. Data Preprocessing: I handled missing values in the Age and Embarked columns and defined categorical and numerical columns for preprocessing.
3. Model Building: I built a pipeline with a ColumnTransformer for preprocessing and a LogisticRegression classifier.
4. Model Training: I trained the model using the training data and evaluated its performance on the test data.
5. Model Evaluation: I calculated the accuracy and classification report for the model and plotted the ROC curve to visualize its performance.

# Installation
To run this code, i need to install the following libraries:

- pandas
- scikit-learn
- matplotlib
- seaborn

 Install these libraries using pip:
 
    pip install pandas scikit-learn matplotlib seaborn


bash
pip install pandas scikit-learn matplotlib seaborn
