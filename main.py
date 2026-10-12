"""
Thiranex Task: Predictive Modeling Using Machine Learning
Example project: predict an exam score from hours studied.
Run: python main.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the dataset (keep student_scores.csv in the same folder)
data = pd.read_csv("student_scores.csv")

# 2. Select input feature (X) and value to predict (y)
X = data[["hours_studied"]]
y = data["exam_score"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict scores for test data and evaluate the model
predictions = model.predict(X_test)
print("Predictive Modeling Using Machine Learning")
print("------------------------------------------")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, predictions):.2f}")
print(f"R-squared score: {r2_score(y_test, predictions):.2f}")

# 6. Predict a score for a new number of study hours
hours = float(input("Enter study hours to predict an exam score: "))
new_prediction = model.predict(pd.DataFrame({"hours_studied": [hours]}))[0]
print(f"Predicted exam score for {hours:g} study hours: {new_prediction:.2f}")

# 7. Save a plot showing the data and fitted regression line
plt.scatter(data["hours_studied"], data["exam_score"], label="Actual data")
plt.plot(
    data["hours_studied"],
    model.predict(data[["hours_studied"]]),
    label="Linear regression line"
)
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Exam Score Prediction")
plt.legend()
plt.tight_layout()
plt.savefig("prediction_plot.png", dpi=150)
print("Saved plot as prediction_plot.png")
