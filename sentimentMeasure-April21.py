from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import csv



# Preprocess text (username and link placeholders)

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)



def calcVideoSentiment(dictionary):
    commentSentiments = []
    neg = 0
    neu = 0
    pos = 0
    for comment in dictionary:
        commentSentiments.append(dictionary[comment][0])
    for sent in commentSentiments:
        if "Positive" in sent:
            pos+=1
        elif "Negative" in sent:
            neg += 1
        else:
            neu += 1
    return [neg, neu, pos]



MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
#model.save_pretrained(MODEL)


with open("largeCreatorPROCESSED_8.txt", "r", encoding = "Latin1") as f:
    lines = f.readlines()
 
finalScore = {}
for x in lines:
    finalScore[x] = []
    text = x 
    text1 = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    for i in range(scores.shape[0]):
        l = config.id2label[ranking[i]]
        s = scores[ranking[i]]
        commentScores = f"{i+1}) {l} {np.round(float(s), 4)}"
        finalScore[x].append(commentScores)
        

print(calcVideoSentiment(finalScore))
    


# # TF
# model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
# model.save_pretrained(MODEL)
# text = "Covid cases are increasing fast!"
# encoded_input = tokenizer(text, return_tensors='tf')
# output = model(encoded_input)
# scores = output[0][0].numpy()
# scores = softmax(scores)
# Print labels and scores
