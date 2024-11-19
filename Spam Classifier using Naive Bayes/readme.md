# Spam Email Detection Using Naive Bayes

## Problem Statement

Email spam detection is crucial in today's digital communication to prevent unwanted messages from flooding users' inboxes. The goal of this project is to classify email messages into two categories:
- **Ham**: Legitimate emails.
- **Spam**: Unwanted, potentially harmful emails.

The dataset, **SMS Spam Collection**, contains 5,728 email messages tagged as either "ham" or "spam." Each message includes:
- **v1**: The label (ham or spam).
- **v2**: The raw text of the email.

This project uses the **Naive Bayes classifier** for its simplicity and effectiveness in text classification tasks.

---

## Project Objective

To design and implement a robust spam email classification model using **Naive Bayes** and natural language processing techniques to identify spam emails with high accuracy.

---

## Dataset

- **Source**: SMS Spam Collection dataset
- **Size**: 
  - 5,728 messages
  - 4,360 (76.1%) messages labeled as ham
  - 1,368 (23.9%) messages labeled as spam
- **Structure**:
  - `text`: Email content
  - `spam`: Label (1 for spam, 0 for ham)

---

## Tech Stack

- **Programming Language**: Python
- **Libraries**:
  - `pandas`, `numpy` for data manipulation.
  - `matplotlib`, `seaborn` for data visualization.
  - `nltk` for natural language processing.
  - `scikit-learn` for machine learning.
- **Environment**: Jupyter Notebook

---

## Methodology

### Step 1: Data Preprocessing
1. **Load Dataset**:
   - Imported email data into a Pandas DataFrame.
   - Explored and visualized the data structure, identifying class distributions.
2. **Add Length Feature**:
   - Calculated the length of each email message as a potential feature for spam detection.

### Step 2: Data Cleaning
1. **Remove Punctuation**:
   - Stripped punctuation marks to reduce noise.
2. **Remove Stopwords**:
   - Filtered out common words (e.g., "the," "and") using the NLTK library.
3. **Tokenization**:
   - Split text into individual words for feature extraction.

### Step 3: Feature Extraction
1. **Bag of Words**:
   - Converted preprocessed text into a matrix of token counts using `CountVectorizer`.
2. **TF-IDF**:
   - Applied **Term Frequency-Inverse Document Frequency** to weigh terms based on importance.

### Step 4: Model Training
1. **Split Dataset**:
   - Divided the dataset into training (80%) and testing (20%) subsets.
2. **Train Naive Bayes Classifier**:
   - Used the **Multinomial Naive Bayes** algorithm due to its effectiveness in text classification.

### Step 5: Model Evaluation
1. **Metrics**:
   - Evaluated performance using accuracy, precision, recall, and F1-score.
   - Visualized results with confusion matrices.

---

## Results

1. **Model Performance** (Using Count Vectorizer):
   - **Precision**: 99%
   - **Recall**: 98%
   - **F1-Score**: 99%
   - **Accuracy**: 99%
2. **TF-IDF Performance**:
   - **Precision**: 77%
   - **Recall**: 77%
   - **F1-Score**: 77%
   - **Accuracy**: 77%

---

## Challenges Faced

1. **Imbalanced Dataset**:
   - Spam emails formed a smaller portion of the dataset, requiring careful consideration to avoid biased predictions.
2. **Noisy Data**:
   - Punctuation, stopwords, and unimportant words affected the quality of feature extraction.
3. **Feature Importance**:
   - Balancing the contributions of TF-IDF weights and token counts.

---

## Key Learnings

1. **Preprocessing Techniques**:
   - Importance of removing punctuation and stopwords to improve model accuracy.
2. **Feature Engineering**:
   - Using text length as a feature improved classification.
3. **Text Representation**:
   - Understanding Bag of Words and TF-IDF for effective text classification.
4. **Naive Bayes**:
   - Learned how Naive Bayes efficiently handles sparse matrices from text datasets.

---

## Future Scope

1. **Enhanced Preprocessing**:
   - Explore stemming or lemmatization to further reduce noise in textual data.
2. **Advanced Models**:
   - Implement more sophisticated models such as BERT or LSTMs for spam detection.
3. **Deployment**:
   - Deploy the model as a real-time email spam filter using Flask or FastAPI.
4. **Multilingual Support**:
   - Extend the model to classify emails in multiple languages.

---

## How to Run

1. **Set up Environment**:
   - Install required libraries:
     ```bash
     pip install pandas numpy matplotlib seaborn nltk scikit-learn
     ```
2. **Run Jupyter Notebook**:
   - Open and execute the provided notebook to preprocess data, train the model, and evaluate performance.

3. **Test with Custom Emails**:
   - Modify the `testing_sample` variable in the notebook to test new email samples.

---

## Conclusion

The **Spam Email Detection Using Naive Bayes** project successfully demonstrates how text preprocessing and a simple classification algorithm can effectively identify spam emails. While the current implementation achieves high accuracy, future improvements in preprocessing and model sophistication can further enhance its capabilities.

