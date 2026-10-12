# Predictive Modeling Using Machine Learning

## Project overview
This beginner-friendly project uses **Linear Regression** to predict an exam score from the number of hours studied. The dataset is included as `student_scores.csv`.

## Files
- `main.py` — loads data, trains and tests the model, prints evaluation metrics, asks for study hours, and predicts a score.
- `student_scores.csv` — sample dataset with study hours and exam scores.
- `requirements.txt` — Python libraries needed to run the project.
- `README.md` — project instructions.

## How to run
1. Install Python 3.
2. Put all project files in the same folder.
3. Open a terminal in that folder and install the libraries:
   ```bash
   pip install -r requirements.txt
   ```
4. Run:
   ```bash
   python main.py
   ```
5. Enter a number of study hours when prompted. The program prints the predicted score and saves `prediction_plot.png`.

## Method
1. Load the CSV data using pandas.
2. Split the data into training and testing sets.
3. Train a Linear Regression model.
4. Evaluate it using Mean Absolute Error (MAE) and R-squared.
5. Predict a score for a new input.

## Note
The included data is **sample educational data**, not real student records. Results are for demonstrating the machine-learning workflow only.
