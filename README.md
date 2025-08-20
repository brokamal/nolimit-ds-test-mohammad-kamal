# Rotten Tomatoes Movie Review Sentiment Classifier

## About
This app predicts the sentiment of movie reviews. It also gives you the closest review in similarty search. You can use the deployed version of this [app](https://huggingface.co/spaces/kamalbdg/distillBERT-RT-Review)
- Model used: [disitillBERT](https://huggingface.co/distilbert/distilbert-base-uncased) 
- Dataset used: [dataset](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes) , License: unknown
- Framework and deploy: [Flask + huggingface space](https://huggingface.co/spaces/kamalbdg/distillBERT-RT-Review)


# Usage 
1. Enter a review of a movie, for example:
- Positive review
```
Classy, arty, horror. Good enough for genre fans. Mature enough for grownups. More than enough.
```
- Negative Review 
```
The tonal shifts don’t always work. Yes, it comes apart. Defiantly so. Then again, the film doesn’t seem to care about narrative precision. It’s about dread, decay, and the unavoidable march of time.`
```

2. Click the button after filling the review.


3. See prediction and similarity search result : 

- Positive result:
![result-poz](/docs/pozreview.png)


- Negative Result
![result-neg](/docs/negreview.png)


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
4. go to the address
```
http://127.0.0.1:7860/predict
```

# Flowchart 
## Training Flowchart
![train-flow](/docs/train.png)
### Explanation 
1. Rotten tomatoes review sentiment dataset is loaded as the dataset
2. Dataset is then preprocessed (tokenazation and padding)
3. distillBERT model is trained using the dataset 
4. Fine-tuned model is evaluated using evaluation metrics. 
5. Re-train if the accuracy is low.
6. Save model if the accuracy is high, as the final fine-tuned model.



## Inference Flowchart
![inf-flow](/docs/inference.png)
### Explanation
1. User input review in form of text.
2. User input is tokenized before being fed into the fine-tuned model.
3. Model receive the input, model perform two tasks: classification and embedding extraction.
4. Classification :  Logits->argmax-> sentiment label (Positive / Negative).
5. Embedding extraction : Hidden state of [CLS] token is taken as the sentence embedding.
6. Similarity search using KNN : The query embedding compared with training set embedding using KNN. Retrives the 5 closest distance of review.
7. Final output for classification is label: positive or Negative
8. Final output for similarity search is the top 5 closest distance review with the distance itself.





