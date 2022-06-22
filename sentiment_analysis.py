from transformers import BertTokenizer
import torch
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F

from aux_classes import *

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

device = torch.device("cpu")

PRE_TRAINED_MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

df = pd.read_csv("new_reviews.csv")

classes = ['negative', 'neutral', 'positive']

def to_sentiment(rating):
    rating = int(rating)
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else: 
        return 2

df['sentiment'] = df.score.apply(to_sentiment)

# Achar comprimento máximo
token_lens = []
for txt in df.content:
  tokens = tokenizer.encode(str(txt), max_length=512, padding='max_length')
  token_lens.append(len(tokens))

MAX_LEN = max(token_lens)

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(
    reviews=df.content.to_numpy(),
    targets=df.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
    )
    return DataLoader(ds,
    batch_size=batch_size,
    num_workers=2
  )

BATCH_SIZE = 32

data_loader = create_data_loader(df, tokenizer, MAX_LEN, BATCH_SIZE)

data = next(iter(data_loader))

input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)

model = SentimentClassifier(3)
model.load_state_dict(torch.load('best_model_state.bin',map_location=torch.device('cpu')))
model = model.to(device)

def get_predictions(model, data_loader):
    model = model.eval()
    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            texts = d["review_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            probs = F.softmax(outputs, dim=1)

            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()

    return review_texts, predictions, prediction_probs, real_values

review_texts, predictions, prediction_probs, real_values = get_predictions(model,data_loader)

for rev,pred,prob,val in zip(review_texts, predictions, prediction_probs, real_values):
    print()
    print("Review: " + str(rev))
    print("Prediction: " + classes[int(pred)])
    #print("Probability: " + str(prob))
    print("Real value: " + classes[int(val)])