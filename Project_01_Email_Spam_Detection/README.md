# 📧 Email Spam Detector


A machine learning-powered web application that classifies email messages as **Spam** or **Not Spam** using Natural Language Processing (NLP).


The project was developed as **Project 01** of the Hex Software Machine Learning Internship.


---


## 🚀 Project Overview


Spam emails are unwanted messages that may contain advertisements, scams, malicious links, or fraudulent offers.


This project builds a machine learning classifier capable of analyzing the text of an email and predicting whether it is:


- 🚨 **Spam**
- ✅ **Not Spam**


The system combines **TF-IDF text vectorization** with a **Multinomial Naive Bayes** classifier and provides a Flask-based web interface for testing new messages.


---


## 🎯 Objectives


The main objectives of this project were to:


- Build an NLP-based spam classification model
- Clean and preprocess raw message data
- Convert text into numerical features using TF-IDF
- Train a Multinomial Naive Bayes classifier
- Compare different model configurations
- Evaluate model performance using multiple metrics
- Build a Flask web application for real-time predictions
- Automatically save prediction results as screenshots


---


## 🛠️ Technologies Used


- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Multinomial Naive Bayes
- Flask
- Joblib
- Pillow
- Matplotlib / Seaborn
- Jupyter Notebook


---


## 📂 Project Structure


```text
Project_01_Email_Spam_Detection/
│
├── data/
│   └── spam.csv
│
├── models/
│   ├── spam_classifier.joblib
│   └── tfidf_vectorizer.joblib
│
├── screenshots/
│   ├── test_001.png
│   ├── test_002.png
│   └── ...
│
├── src/
│   ├── preprocess.py
│   └── model_experiments.py
│
├── templates/
│   └── index.html
│
├── app.py
├── train.py
├── requirements.txt
└── README.md
📊 Dataset

The project uses the SMS Spam Collection dataset obtained through KaggleHub.

The original dataset contains labeled messages with two main categories:

ham
spam

The data was cleaned and transformed into the project's classification format:

0 → Not Spam
1 → Spam
Dataset statistics

Original dataset:

5,572 messages
4,825 ham messages
747 spam messages

After preprocessing:

5,169 usable messages
4,516 Not Spam
653 Spam

The dataset is imbalanced, with substantially more legitimate messages than spam messages.

🧹 Data Preprocessing

The preprocessing pipeline:

Loaded the raw CSV dataset
Selected the message and label columns
Removed unnecessary columns
Removed missing values
Converted labels into numerical values
Prepared the cleaned dataset for machine learning

The preprocessing script is located at:

src/preprocess.py
🔤 Feature Engineering

Machine learning models cannot directly process raw text.

The project therefore uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert email text into numerical feature vectors.

TF-IDF assigns greater importance to words that are useful for distinguishing between spam and legitimate messages while reducing the importance of very common words.

The resulting feature matrix is passed to the machine learning classifier.

🤖 Machine Learning Model

The main classifier used is:

Multinomial Naive Bayes

Multinomial Naive Bayes is well suited for text classification because it works effectively with word-frequency-based features such as TF-IDF.

The model learns patterns associated with spam and legitimate messages and uses those patterns to classify previously unseen messages.

🧪 Model Experiments

Several configurations were evaluated.

Experiment	Accuracy	Precision	Recall	F1
Baseline	95.36%	100.00%	63.36%	77.57%
Higher Alpha	96.71%	98.99%	74.81%	85.22%
Lower Alpha	98.45%	97.52%	90.08%	93.65%
Unigrams + Bigrams + Trigrams	96.23%	100.00%	70.23%	82.51%

The Lower Alpha configuration produced the strongest overall performance.

📈 Final Model Performance

The final model achieved:

Accuracy: 98.45%
Classification Report
Class	Precision	Recall	F1-score
Not Spam	99%	100%	99%
Spam	98%	90%	94%
Confusion Matrix
                Predicted
                Not Spam   Spam


Actual Not Spam    900       3
Actual Spam         13      118

The model correctly identified 118 of 131 spam messages in the test set while producing only 3 false spam predictions for legitimate messages.

🌐 Flask Web Application

A Flask web interface was developed to allow users to enter an email message and receive a prediction.

The application displays:

Email input field
Character counter
Spam / Not Spam prediction
Model confidence
Recent prediction history
Clear history functionality

Run the application with:

python app.py

Then open:

http://127.0.0.1:5000
📸 Automatic Prediction Screenshots

Each prediction is automatically saved by the Flask backend as a PNG image.

Screenshots are stored directly in:

screenshots/

Example:

screenshots/
├── test_001.png
├── test_002.png
├── test_003.png
└── ...

This provides visual evidence of the application's predictions without requiring manual screenshots.

🧪 Example Test Cases
Example 1 — Spam
Congratulations! You have been selected to win a $1,000 prize.
Call now to claim your reward. Limited time offer!

Expected result:

🚨 Spam
Example 2 — Not Spam
Hi Sarah,


Just checking if we're still meeting for lunch tomorrow.
Let me know what time works for you.


Thanks!

Expected result:

✅ Not Spam
Example 3 — Spam
WINNER! You have won a FREE cash prize.
Claim your reward now by calling the number below.
Offer expires today!

Expected result:

🚨 Spam
Example 4 — Not Spam
Hello John,


Please find the meeting notes attached.
Let me know if you have any questions.


Best regards,
David

Expected result:

✅ Not Spam
💾 Saved Model Files

After training, the following files are generated:

models/spam_classifier.joblib
models/tfidf_vectorizer.joblib

These allow the Flask application to load the trained model without retraining every time the application starts.

⚙️ Installation

Clone the repository and navigate to the project:

cd Project_01_Email_Spam_Detection

Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt
▶️ Training the Model

Run preprocessing:

python src/preprocess.py

Train the model:

python train.py

Run the model experiments:

python src/model_experiments.py
🌐 Running the Application

Start Flask:

python app.py

Open the application in your browser:

http://127.0.0.1:5000

Enter an email message and click:

🔍 Analyze Email

🔮 Future Improvements

Possible improvements include:

Training with a larger and more diverse email spam dataset
Supporting full email classification including subject, sender, and metadata
Comparing Naive Bayes with Logistic Regression, SVM, and other classifiers
Improving text preprocessing and normalization
Deploying the Flask application
Adding a prediction history dashboard
Improving the user interface
Adding automated model monitoring
Handling HTML email content
Testing the model against real-world spam campaigns
Improving detection of long-form emails
📌 Key Learning Outcomes

Through this project, I gained practical experience in:

Natural Language Processing
Text preprocessing
TF-IDF feature engineering
Naive Bayes classification
Model evaluation
Hyperparameter experimentation
Confusion matrix analysis
Flask application development
Model serialization with Joblib
Integrating machine learning models into web applications
Generating automated prediction evidence
👩‍💻 Internship Project

Hex Software — Machine Learning Internship

Project 01: Email Spam Detector

Built using Python, NLP, Scikit-learn, Multinomial Naive Bayes, and Flask.