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
The file is in the .iypnb (Jupyter Lab) format. Open and run the code in Visual Studio Code (or a similar program). Acquire the appropriate .iypnb. 
Execute each section of code starting from the top to the end. 
Several sections of code are grouped together into these respective areas: 

### 1. **Import Libraries**
- Import necessary libraries for data handling, preprocessing, and model building.

### 2. **Load Dataset**
- Read the CSV files (`train.csv` and `test.csv`).
- Separate features (`X`) and target (`y`).

### 3. **Data Preprocessing**
- Clean the text data by:
  - Removing non-alphabetic characters.
  - Lowercasing all text.
  - Tokenizing and removing stopwords.
  - Applying lemmatization for normalization.

### 4. **Split Dataset**
- Divide the dataset into training and validation sets using `train_test_split`.
- Stratify the split to maintain the class distribution.

### 5. **Tokenization and Padding**
- Tokenize the cleaned text using Keras `Tokenizer`.
- Convert the tokenized text into padded sequences.

### 6. **Build and Train Machine Models** 
- **Train Naive Bayes Model**
  - Create a baseline machine learning model using `MultinomialNB`.
  - Vectorize the cleaned text using `CountVectorizer` for the Naive Bayes model.
  - Train the model on the training data and evaluate on validation data.
- **Train Decision Tree Model**
  - Insert info. 

### 7. **Build and Train LSTM Model**
- Create an LSTM-based deep learning model using Keras.
- Use `Embedding` for word representations.
- Compile the model with `binary_crossentropy` loss to handle binary classification.
- Train the model on the tokenized and padded sequences.

### 8. **Build and Train BERT Model**
- Insert info. 

### 9. **Evaluate Models**
- Compare the accuracies of each model
- Output metrics like accuracy, precision, recall, and F1-score.
