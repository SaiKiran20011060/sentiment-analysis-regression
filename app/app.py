import streamlit as st
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords')

# ------------------------------
# PAGE CONFIG
# ------------------------------

st.set_page_config(
    page_title="Flipkart Sentiment Analysis",
    layout="wide"
)

st.title("Flipkart Customer Sentiment Analysis")
st.write("Custom Rule-Based Sentiment Analysis using Random Forest")

# ------------------------------
# PREDEFINED WORD LISTS
# ------------------------------

positive_words = [
    "good", "excellent", "amazing", "awesome",
    "best", "love", "perfect", "great",
    "nice", "wonderful", "fantastic", "super",
    "happy", "satisfied", "worth"
]

negative_words = [
    "bad", "worst", "poor", "terrible",
    "hate", "waste", "broken", "slow",
    "disappointed", "problem", "issue",
    "damage", "useless", "boring"
]

neutral_words = [
    "okay", "average", "normal",
    "fine", "medium", "standard"
]

# ------------------------------
# FILE UPLOAD
# ------------------------------

uploaded_file = st.file_uploader(
    "Upload Flipkart CSV File",
    type=["csv"]
)

# ------------------------------
# MAIN APP
# ------------------------------

if uploaded_file is not None:

    # ------------------------------
    # FILE SIZE VALIDATION
    # ------------------------------

    file_size_mb = uploaded_file.size / (1024 * 1024)

    if file_size_mb > 200:

        st.error("File size exceeds 200 MB. Please upload a smaller CSV file.")
        st.stop()

    try:

        # ------------------------------
        # READ CSV
        # ------------------------------

        df = pd.read_csv(uploaded_file)

        # ------------------------------
        # REQUIRED COLUMN CHECK
        # ------------------------------

        required_columns = ['Review', 'Rating']

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:

            st.error(
                f"Dataset does not contain required column(s): {', '.join(missing_columns)}"
            )

            st.info("Required columns are: Review, Rating")

            st.stop()

        # ------------------------------
        # DATA PREVIEW
        # ------------------------------

        st.subheader("Dataset Preview")

        st.dataframe(df.head())

        st.success(f"File uploaded successfully ({file_size_mb:.2f} MB)")

        # ------------------------------
        # CLEAN DATA
        # ------------------------------

        df.dropna(inplace=True)

        df['Review'] = df['Review'].astype(str)

        stop_words = set(stopwords.words('english'))

        def clean_text(text):

            text = text.lower()

            text = re.sub(r'[^a-zA-Z]', ' ', text)

            words = text.split()

            words = [word for word in words if word not in stop_words]

            return words

        # ------------------------------
        # CUSTOM FEATURE EXTRACTION
        # ------------------------------

        def count_sentiment_words(review):

            words = clean_text(review)

            pos_count = 0
            neg_count = 0
            neu_count = 0

            for word in words:

                if word in positive_words:
                    pos_count += 1

                elif word in negative_words:
                    neg_count += 1

                elif word in neutral_words:
                    neu_count += 1

            return pd.Series([pos_count, neg_count, neu_count])

        df[['Positive_Count', 'Negative_Count', 'Neutral_Count']] = df['Review'].apply(
            count_sentiment_words
        )

        # ------------------------------
        # CREATE TARGET LABELS
        # ------------------------------

        def get_sentiment(rating):

            if rating >= 4:
                return 'Positive'

            elif rating == 3:
                return 'Neutral'

            else:
                return 'Negative'

        df['Sentiment'] = df['Rating'].apply(get_sentiment)

        # ------------------------------
        # FEATURES & TARGET
        # ------------------------------

        X = df[['Positive_Count', 'Negative_Count', 'Neutral_Count']]

        y = df['Sentiment']

        # ------------------------------
        # TRAIN TEST SPLIT
        # ------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # ------------------------------
        # RANDOM FOREST MODEL
        # ------------------------------

        model = RandomForestClassifier()

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        # ------------------------------
        # MODEL ACCURACY
        # ------------------------------

        st.subheader("Model Accuracy")

        st.success(f"Random Forest Accuracy: {accuracy:.2f}")

        # ------------------------------
        # SENTIMENT DISTRIBUTION
        # ------------------------------

        st.subheader("Dataset Sentiment Distribution")

        sentiment_counts = df['Sentiment'].value_counts()

        col1, col2, col3 = st.columns(3)

        col1.metric("Positive", sentiment_counts.get('Positive', 0))
        col2.metric("Neutral", sentiment_counts.get('Neutral', 0))
        col3.metric("Negative", sentiment_counts.get('Negative', 0))

        st.bar_chart(sentiment_counts)

        # ------------------------------
        # USER REVIEW PREDICTION
        # ------------------------------

        st.subheader("Predict Customer Review")

        user_review = st.text_area("Enter Review Text")

        if st.button("Predict Sentiment"):

            if user_review.strip() == "":

                st.warning("Please enter a review.")

            else:

                words = clean_text(user_review)

                pos_count = 0
                neg_count = 0
                neu_count = 0

                matched_positive = []
                matched_negative = []
                matched_neutral = []

                for word in words:

                    if word in positive_words:
                        pos_count += 1
                        matched_positive.append(word)

                    elif word in negative_words:
                        neg_count += 1
                        matched_negative.append(word)

                    elif word in neutral_words:
                        neu_count += 1
                        matched_neutral.append(word)

                input_data = pd.DataFrame({
                    'Positive_Count': [pos_count],
                    'Negative_Count': [neg_count],
                    'Neutral_Count': [neu_count]
                })

                prediction = model.predict(input_data)[0]

                # ------------------------------
                # DETECTED WORDS
                # ------------------------------

                st.subheader("Detected Words")

                st.write("### Positive Words")
                st.write(
                    matched_positive
                    if matched_positive
                    else "No positive words found"
                )

                st.write("### Negative Words")
                st.write(
                    matched_negative
                    if matched_negative
                    else "No negative words found"
                )

                st.write("### Neutral Words")
                st.write(
                    matched_neutral
                    if matched_neutral
                    else "No neutral words found"
                )

                # ------------------------------
                # FINAL PREDICTION
                # ------------------------------

                st.subheader("Final Prediction")

                if prediction == 'Positive':

                    st.success(f"Predicted Sentiment: {prediction}")

                elif prediction == 'Negative':

                    st.error(f"Predicted Sentiment: {prediction}")

                elif prediction == 'Neutral':

                    st.info(f"Predicted Sentiment: {prediction}")

                else:

                    st.warning(f"Predicted Sentiment: {prediction}")

    except Exception:

        st.error("Unable to process the uploaded file.")

        st.info("Please upload a valid CSV file containing Review and Rating columns.")

else:

    st.info("Please upload a CSV file to continue.")