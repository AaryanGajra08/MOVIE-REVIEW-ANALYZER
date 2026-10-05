MOVIE REVIEW ANALYSIS
======================
A complete academic NLP project using an IMDb-trained sentiment model.

The included model was trained on 50,000 IMDb reviews using:
- Text cleaning
- TF-IDF (unigrams + bigrams)
- Logistic Regression
- 80/20 stratified train/test split

Test accuracy from the training run: 91.00%

MAIN COMPONENTS
---------------
1. model/imdb_sentiment_model.joblib
   The trained sentiment model.

2. app.py
   FastAPI backend with:
   GET  /health
   POST /predict
   POST /batch-predict
   GET  /docs

3. frontend/index.html
   Website connected to the real API.

4. model/train_model.py
   Re-trains the model using data/IMDB Dataset.csv.

5. data/sample_reviews.csv
   Small batch-analysis demo.

6. demo.py
   Command-line API demo.

7. STEP_BY_STEP.txt
   Full Windows + VS Code setup instructions.

DEMO
----
Start the API:
    python -m uvicorn app:app --reload

Then open:
    frontend/index.html

Type:
    The acting was excellent and the story was fantastic. I loved this movie.

Click Analyze Review.

The website sends the review to FastAPI, which loads the trained .joblib model and returns sentiment and confidence.

NOTE
----
The 91.00% figure is the test accuracy from the supplied IMDb dataset and the specific training run. It is not a guarantee of accuracy on every new movie review.
