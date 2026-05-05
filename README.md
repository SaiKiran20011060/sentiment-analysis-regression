# Flipkart Sentiment Analysis

A machine learning-based application for analyzing customer sentiment from Flipkart product reviews using Random Forest classification.

## Overview

This project implements a custom rule-based sentiment analysis system combined with machine learning to classify customer reviews as **Positive**, **Neutral**, or **Negative**. The application uses a Random Forest classifier trained on predefined sentiment word lists and custom feature extraction.

## Features

- **CSV File Upload**: Upload Flipkart review data (CSV format)
- **Sentiment Classification**: Classify reviews into three categories (Positive, Neutral, Negative)
- **Rule-Based Feature Extraction**: Extract sentiment features based on predefined word lists
- **Model Training**: Automatically train Random Forest model on uploaded data
- **Accuracy Metrics**: Display model accuracy and sentiment distribution
- **Review Prediction**: Predict sentiment for custom user-entered reviews
- **Word Detection**: Identify sentiment-bearing words in reviews

## Project Structure

```
flipkart_sentiment_analysis/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── app/
│   └── app.py                  # Streamlit application
├── data/
│   └── flipkart.csv            # Sample data (replace with your CSV)
└── src/
    └── sentiment_Analyser.ipynb # Jupyter notebook for analysis
```

## Requirements

- Python 3.7+
- pandas
- scikit-learn
- nltk
- streamlit

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd flipkart_sentiment_analysis
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # Activate on Windows
   venv\Scripts\activate
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

### Using Streamlit (Web Interface)

```bash
streamlit run app/app.py
```

This will launch the web application in your default browser at `http://localhost:8501`

### Using Jupyter Notebook

```bash
jupyter notebook src/sentiment_Analyser.ipynb
```

## Usage

1. **Prepare Your Data**:
   - Create a CSV file with columns: `Review` and `Rating`
   - Rating should be numeric (1-5 recommended)

2. **Upload Data**:
   - Open the Streamlit app
   - Click "Upload Flipkart CSV File"
   - Select your CSV file

3. **View Results**:
   - Dataset preview and file size confirmation
   - Model accuracy percentage
   - Sentiment distribution chart
   - Metrics showing count of Positive, Neutral, and Negative sentiments

4. **Predict Sentiment**:
   - Enter custom review text in the text area
   - Click "Predict Sentiment" button
   - View detected words and final prediction

## Data Format

Your CSV file should contain at least these columns:

| Column | Type | Description |
|--------|------|-------------|
| Review | String | Customer review text |
| Rating | Integer | Rating (1-5) |

Example:
```
Review,Rating
"Great product works well",5
"Not satisfied with quality",2
"It is okay",3
```

## Model Performance

The application uses:
- **Algorithm**: Random Forest Classifier
- **Features**: Count of Positive, Negative, and Neutral words
- **Train-Test Split**: 80-20 ratio
- **Random State**: 42 (for reproducibility)

## Sentiment Classification Rules

- **Positive**: Rating ≥ 4
- **Neutral**: Rating = 3
- **Negative**: Rating < 3

## Predefined Sentiment Words

### Positive Words
good, excellent, amazing, awesome, best, love, perfect, great, nice, wonderful, fantastic, super, happy, satisfied, worth

### Negative Words
bad, worst, poor, terrible, hate, waste, broken, slow, disappointed, problem, issue, damage, useless, boring

### Neutral Words
okay, average, normal, fine, medium, standard

## Limitations

- Maximum file size: 200 MB
- Predefined word lists are English-only
- Sentiment is classified based on rating, not review content alone
- Custom words can be added by modifying the word lists in `app.py`

## Troubleshooting

**"File size exceeds 200 MB"**
- Upload a smaller CSV file with fewer rows

**"Dataset does not contain required column(s)"**
- Ensure your CSV has `Review` and `Rating` columns

**"Unable to process the uploaded file"**
- Verify your CSV format is valid
- Check that data types match expected format

## Future Enhancements

- Add support for other languages
- Implement TF-IDF vectorization
- Add more sophisticated NLP preprocessing
- Include model export/save functionality
- Add sentiment confidence scores

## License

This project is provided as-is for educational purposes.

## Author

Flipkart Sentiment Analysis Project

## Support

For issues or questions, please check the code documentation or the Jupyter notebook for detailed explanations of the analysis process.
