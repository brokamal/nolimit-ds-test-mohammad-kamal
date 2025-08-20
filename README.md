# Rotten Tomatoes Movie Review Sentiment Classifier

## About
This app predicts the sentiment of movie reviews. disitillBERT was fine tuned using Rotten Tomatoes Review dataset. You can use the deployed version of this [app](https://huggingface.co/spaces/kamalbdg/distillBERT-RT-Review)
- Model used: [disitillBERT](https://huggingface.co/distilbert/distilbert-base-uncased) 
- Dataset used: [dataset](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes) , License: MIT
- Framework and deploy: [Flask + huggingface space](https://huggingface.co/spaces/kamalbdg/distillBERT-RT-Review)


# Usage 
1. Enter a review of a movies, for example:
```
The tonal shifts don’t always work. Yes, it comes apart. Defiantly so. Then again, the film doesn’t seem to care about narrative precision. It’s about dread, decay, and the unavoidable march of time.`
```
2. Click the button after filling the review.


3. See prediction result 

- Positive result:
![result-poz](/docs/poz.png)

- Negative Result
![result-neg](/docs/poz.png)


# How to run
Here's how you can run the app locally:

1. clone this respository 
```
git clone https://github.com/brokamal/nolimit-ds-test-mohammad-kamal
```
2. install dependencies
```
pip3 install -r requirements.txt
```
3. run app.py
```
python3 app.py
```
3. go to the address
```
http://127.0.0.1:7860/predict
```

# Flowchart 







# License
MIT
