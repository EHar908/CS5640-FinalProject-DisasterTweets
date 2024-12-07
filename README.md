# CS5640-FinalProject-DisasterTweets
Final Code Submission

Dependencies:
- Libraries:
  - pandas
  - numpy
  - sklearn
    - train_test_split
    - classification_report
    - compute_class_weight
    - resample
  - tensorflow
    - keras
      - Sequential
      - Embedding
      - LSTM
      - Dense
      - Dropout
      - Bidirectional
      - Tokenizer
      - pad-sequences
  - re (Regular Expressions)
  - nltk
    - stopwords
    - WordNetLemmatizer
    - wordnet

Instructions: 
The file is in the .iypnb (Jupyter Lab) format. Open and run the code in Visual Studio Code (or a similar program). Acquire the appropriate .iypnb extensions to run it. 
Execute each section of code starting from the top to the end. 
Several portions of code are grouped together into these respective sections: 

### 1. **Import Libraries**
- Import necessary libraries for data handling, preprocessing, and model building.

### 2. **Classification Reports List**
- Create an empty array for storing classification reports of each model. 

### 3. **Load Dataset**
- Read the CSV files (`train.csv` and `test.csv`).

### 4. **Decision Tree**
- Data Pre-processing:
   - Removes URLs.
   - Removes mentions (e.g., @username).
   - Removes hashtags (#).
   - Removes special characters.
   - Removes extra spaces.
   - Fills missing values in the keyword, location, and text columns with empty strings.
   - Cleans the text column using the clean_text function.
   - Combines the text, repeated keyword (3 times), and location into a single combined column.
- Data Splitting
 - Splitting features and target
 - Stratified trian-validation split
- Build and train model
  - Compute class weights for imbalanced data  
  - Define pipeline with Decision Tree  
  - Hyperparameter tuning with GridSearchCV  
  - Best model from grid search
- Evaluate and Store Classification Report in Reports Array

### 5. **Naive Bayes**
- Data Pre-processing:
 - Drop unnecessary columns and handle missing values
 - Resample data to handle class imbalance
 - Split data into features (X) and target (y)
 - Initialize lemmatizer and stopword set
 - Preprocessing function
 - Remove non-alphabetic characters
 - Convert text to lowercase and split into words
 - Remove stopwords and apply lemmatization
 - Apply the preprocessing function to the entire dataset
- Data Splitting:
 - Split into training and avlidation sets
- Tokenize and Pad Sequences:
 - Initialize tokenizer with a vocabulary size of 10,000 and an out-of-vocabulary token (<OOV>)
 - Fit the tokenizer on the training data (X_train)
 - Convert training and validation texts to sequences of integers
 - Pad and truncate training sequences to a maximum length of 100 using post-padding and post-truncating
 - Pad and truncate validation sequences to a maximum length of 100 using post-padding and post-truncating
- Build and Train:
 - Get class weights for imbalanced data
 - Define TF-IDF Vectorizer and Naive Bayes pipeline
 - Fit the pipeline
- Evaluate and Store Classification Report in Reports Array

### 6. **Build and Train LSTM Model**
- Create an LSTM-based deep learning model using Keras.
- Use `Embedding` for word representations.
- Compile the model with `binary_crossentropy` loss to handle binary classification.
- Train the model on the tokenized and padded sequences.

### 7. **Build and Train BERT Model**
- Insert info. 

### 8. **Evaluate Models**
- Compare the accuracies of each model
- Output metrics like accuracy, precision, recall, and F1-score.
