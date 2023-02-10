import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import warnings
warnings.filterwarnings('ignore')

import torch
import pandas as pd
import random
import math
import numpy as np
import nltk
from sklearn.metrics import classification_report
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import transformers
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
import pickle
import torch.optim as optim
from sklearn import preprocessing
import wandb
torch.cuda.is_available()
import logging
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# !pip install trl
# import trl.gpt2
import torch
import wandb
import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
tqdm.pandas()
from datasets import load_dataset

from transformers import GPT2Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import trl
from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
from trl.core import build_bert_batch_from_txt
def concat_ (a,b):
    new = []
    for i in range(len (a)):
        new.append(str(a[i])+': '+str(b[i]))
    return new

df1 = pd.read_csv('train_new_final.csv')
df2 = pd.read_csv('test.csv')

# train_final = concat_(df1['Type'].to_list(),df1['Utterance'].to_list())
# test_final = concat_(df2['Type'].to_list(),df2['Utterance'].to_list())

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

df_edited = df1
# df_edited["Dialogue_Act"] = label_encoder.fit_transform(df_edited["Dialogue_Act"])

print("We are working on:",device)

print("======================= WandB Login ===========================")
LOGGING = True
if LOGGING: wandb.init(project="gpt2_dac_response")

print("======================= Reading Data ===========================")

df = df_edited[:-1]
counter = 0
label_encoder = preprocessing.LabelEncoder()
import torch
from torch import nn
from transformers import GPT2Model, GPT2LMHeadModel, AutoModel


from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model, GPT2PreTrainedModel
from transformers import top_k_top_p_filtering
from transformers.modeling_outputs import ModelOutput
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F
import torch
from dataclasses import dataclass
from typing import Optional, Tuple


# df
import sentence_transformers

from sentence_transformers import SentenceTransformer, util

import re
import string
import nltk
nltk.download('omw-1.4')

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()

def clean_sentence(question):
  question = question+ " "
  question = question.lower()
  question = question.replace(" tab ", " tablet ")
  # question = question.replace(" tab", " tablet")
  question = question.replace(" cap ", " capsule ")
  question = question.replace(" inj ", " injection ")
  # question = question.replace(" inj", " injection")
  # question = question.replace(" inj.", " injection.")
  text = question
  specials = ["’", "‘", "´", "`"]
  for s in specials:
      text = text.replace(s, "'")
    #Remove Punctuations
  text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
  text = re.sub(r"([?.!,¿])", r" \1 ", text)
  text = re.sub(r'[" "]+', " ", text)
  question = text


  question = re.sub('<[^>]*>', ' ',question)
  question = re.sub(' +', ' ', question)
  question = re.sub('\xa0','',question)
  question = question.rstrip()
  question = re.sub('nan','',question)
  question = re.sub(u'\u2004','',question)
  question = re.sub(u'\u2009','',question)

  # question = question.decode("utf-8")
  # question = question.replace(u'\u200\d*','').encode("utf-8")
  question = re.sub('&nbsp','',question)
  question = re.sub('&ndash','',question)
  question = re.sub('\r','',question)
  question = re.sub('\t','',question)
  question = re.sub('\n',' ',question)

  question = re.sub('MathType@.*','',question)
  question = re.sub('&thinsp','',question)
  question = re.sub('&times','',question)
  question = re.sub('\u200b','',question)
  question = re.sub('&rarr;;;','',question)
  
  question = lemmatizer.lemmatize(question)
  return question



print("======================= WandB Login ===========================")
LOGGING = True
if LOGGING: wandb.init(project="gpt2_dac_response")


model = GPT2LMHeadModel.from_pretrained("gpt2")

gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<sos>', eos_token='<eos>', pad_token='<pad>', truncation=True)
model.resize_token_embeddings(len(gpt2_tokenizer))

model.load_state_dict(torch.load("model_ref.pt")) 


model.save_pretrained("GPT2_base")


model.load_state_dict(torch.load("model_ref.pt")) 


model.save_pretrained("GPT2")



gpt2_model = GPT2HeadWithValueModel.from_pretrained("GPT2_base", local_files_only = True)
gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained("GPT2", local_files_only = True )
# gpt2_tokenizer = GPT2Tokenizer.from_pretrained("GPT2Tokenizer")
print(gpt2_tokenizer.pad_token_id)
gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))
gpt2_model_ref.resize_token_embeddings(len(gpt2_tokenizer))
# gpt2_model.load_state_dict(torch.load("model_20epoch.pt")) 

# gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', truncation=True)
# gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
# gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))
# gpt2_model_ref.resize_token_embeddings(len(gpt2_tokenizer))
wandb.watch(gpt2_model, log='all')    


    
config = {
    "tk_name": "gpt2",
    "steps": 20000,
    "batch_size": 16,
    "forward_batch_size": 8,
    "ppo_epochs": 4,   
    "txt_in_len": 15,
    "txt_out_len": 50,
    "lr": 1.41e-6,
    "init_kl_coef":0.2,
    "target": 6,
    "horizon":1000,
    "gamma":0.99,
    "lam":0.95,
    "cliprange": .2,
    "cliprange_value":.2,
    "vf_coef":.1, 
}
# gpt2_model= nn.DataParallel(model)
# gpt2_model_ref= nn.DataParallel(model)

_ = gpt2_model.to(device)
_ = gpt2_model_ref.to(device)

import nltk
import evaluate 
import rouge
nltk.download('punkt')

model_sentence = SentenceTransformer('paraphrase-MiniLM-L3-v2')



df3 = pd.read_csv("s_train.csv")
df["target"] = [str(x) for x in df3["Utterance"][1:]]



# df["next"] = [">>"+str(x)+": " for x in df3.Type[1:]]
# df.Utterance = df.Utterance + df.next
# df.Utternace = df.Utterance.apply(clean_sentence)

### ADDING CLASSIFIER
# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model. 
# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model. 
from transformers import AutoModel
class DistillBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistillBERTClass, self).__init__()
        self.l1 = AutoModel.from_pretrained("microsoft/xtremedistil-l12-h384-uncased")
        self.pre_classifier = torch.nn.Linear(384, 384)
        self.dropout = torch.nn.Dropout(0.3)
        self.l2 = torch.nn.Linear(384, 200)
        self.l3 = torch.nn.Linear(200, 100)
        self.classifier = torch.nn.Linear(100, 43)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        pooler = self.l2(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.l3(pooler)
        pooler = torch.nn.ReLU()(pooler)
        output = self.classifier(pooler)
        return output
    
model_classifier = DistillBERTClass().to(device)
model_classifier.load_state_dict(torch.load("classifier_new.pt"))
classify_tokenizer = AutoTokenizer.from_pretrained("microsoft/xtremedistil-l12-h384-uncased")


def get_class_logits(sent):
    tokens = torch.tensor(classify_tokenizer.encode_plus(sent)["input_ids"]).unsqueeze(0).to(device)
    masks = torch.tensor(classify_tokenizer.encode_plus(sent)["attention_mask"]).unsqueeze(0).to(device)
    return model_classifier(input_ids = tokens, attention_mask = masks)
    
    
def get_diversity(actual, generated):
    rouge = evaluate.load('rouge')
    results_r = rouge.compute(predictions=[generated],
                         references=[actual])
    
    generated = generated.split(" ")
    actual = actual.split(" ")
    
    results = nltk.translate.bleu_score.sentence_bleu([actual], generated, weights = [1]) 
    return results + results_r["rouge1"] + 4*results_r["rouge2"] + 2*results_r["rougeL"]

def get_reward(model, sentence_1, sentence_2, label):
    lambda1 = 1
    sentences = [sentence_1, sentence_2]
    paraphrase_score = util.paraphrase_mining(model, sentences)[0][0]
    print("Gnerated: ", sentence_1)
    print("Target: ", sentence_2)
    reward = get_class_logits(sentence_1)[0][label]
    diversity =  paraphrase_score + 0.1*reward + get_diversity(sentence_2, sentence_1)
    print(diversity - 0.2)
    # if (1 - (lambda1 * (diversity))) >0.5:
    #     print("reward +1")
    #     return 1.0
    # else: 
    #     print("reward -1")
    #     return -1.0
    return (diversity- 0.2)
    
df["query"] = df.Utterance

labels = [int(x) for x in df["Dialogue_Act"][1:]]
labels.append(0)


 

df["dac_labels"] = labels
# gpt2_tokenizer.padding_side = 'left'
df["tokens"] = [torch.tensor(gpt2_tokenizer.encode_plus(str(x), \
                max_length=200, pad_to_max_length=True, add_special_tokens=True, truncation = True, return_attention_mask=True)["input_ids"]
) for x in df["Utterance"]]


df["masks"] = [torch.tensor(gpt2_tokenizer.encode_plus(str(x), \
                max_length=128, pad_to_max_length=True, add_special_tokens=True, truncation = True, return_attention_mask=True)["attention_mask"]
) for x in df["Utterance"]]

# import 
gen_kwargs = {
    "min_length":-1,
    "top_k": 8,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": gpt2_tokenizer.pad_token_id
}
import warnings
warnings.filterwarnings("ignore")
import re
# from quantulum3 import parser
import pandas as pd
import numpy as np
torch.autograd.set_detect_anomaly(True)
counter = 0

import torch
import logging
import random
import pickle
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
import evaluate 

ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, gpt2_tokenizer,  **config)
fbs = config['forward_batch_size']
gen_len = 10
for epoch in tqdm(range(int(np.ceil(config["steps"]/config['batch_size'])))):
    targs = []
    gens = []
    logs = dict()
    game_data = dict()
    timing = dict()
    t0 = time.time()
    df_batch = df.sample(config['batch_size'])
    game_data['query'] = df_batch['query'].tolist()
    game_data["target"] = df_batch["target"].tolist()
    for i in game_data["target"]:
        targs.append(i)
    tens = []
    for i in df_batch['tokens']:
        tens.append(torch.tensor(i).to(device))
        
    mask = []
    
    for i in df_batch['masks']:
        mask.append(torch.tensor(i).to(device))
    query_tensors = torch.stack(tens)
    mask_tensors = torch.stack(mask)
    dac_labels = torch.tensor(df_batch["dac_labels"].tolist()).to(device)

    t = time.time()
    total_length = config['txt_in_len']+config['txt_out_len']
    response_tensors = []
    for i in range(int(config['batch_size']/fbs)):
        response  = respond_to_batch(gpt2_model, query_tensors[i*fbs:(i+1)*fbs], mask_tensors[i*fbs:(i+1)*fbs], 10)
        response_tensors.append(response)
        # for j in range(i*fbs,(i+1)*fbs):
        # response = gpt2_model.generate(query_tensors[i*fbs:(i+1)*fbs], 
        #                                max_new_tokens=gen_len, **gen_kwargs, size_penalty = -100)
        # response_tensors.append(response.squeeze()[-gen_len:])
        # response_tensors.append(response)
    response_tensors = torch.cat(response_tensors).to(device)
    game_data['response'] = [gpt2_tokenizer.decode(response_tensors[i, :],  skip_special_tokens= False ) for i in range(config['batch_size'])]
    for i in game_data['response']:
        gens.append(i)
        
    timing['time/get_response'] = time.time()-t

    t = time.time()
    timing['time/build_input_sentiment'] = time.time()-t

    t = time.time()
    rewards = []
    for i in range(int(config['batch_size']/fbs)):
        for sentence1, sentence2, label in zip(game_data['response'][i*fbs:(i+1)*fbs],game_data['target'][i*fbs:(i+1)*fbs], dac_labels[i*fbs:(i+1)*fbs]):
            res = get_reward(model_sentence, sentence1, sentence2, label)
            rewards.append(torch.tensor([res]))
    rewards = torch.cat(rewards).to(device)
    timing['time/get_sentiment_preds'] = time.time()-t
    
    #### Run PPO training 
    t = time.time()
    stats = ppo_trainer.step(query_tensors,mask_tensors, response_tensors, rewards, dac_labels)
    timing['time/optimization'] = time.time()-t
     
    #### Log everything
    timing['time/epoch'] = time.time()-t0
    table_rows = [list(r) for r in zip(game_data['query'], game_data['response'], rewards.cpu().tolist())]
    logs.update({'game_log':wandb.Table(
        columns=['query', 'response', 'reward'],
        rows=table_rows)})
    logs.update(timing)
    logs.update(stats)
    logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy()
    logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
    logs['env/reward_dist'] = rewards.cpu().numpy()
    counter += 1
    if counter % 5 == 0:
        name = "model_checkpoint_final" + str(counter) + ".pt"        
        torch.save(gpt2_model.state_dict(), name)
    rouge = evaluate.load('rouge')
    results_r = rouge.compute(predictions=gens,references=targs)
    print(results_r)
    wandb.log(logs)
    
    
torch.save(gpt2_model.state_dict(), "model_final.pt")
# gpt2_model.load_state_dict(torch.load("model_trained.pt")) 
# gpt2_model_ref 

#### get a batch from the dataset
bs = 128
game_data = dict()
df_batch = df.sample(bs)
game_data['query'] = df_batch['query'].tolist()
query_tensors = torch.stack(df_batch['tokens'].tolist()).to(device)

#### get response from gpt2 and gpt2_ref
total_length = config['txt_in_len']+config['txt_out_len']
response_tensors_ref  = gpt2_model.generate(query_tensors)
game_data['Response'] = [gpt2_tokenizer.decode(response_tensors_ref[i, :], skip_special_tokens = True) for i in range(bs)]

response_tensors  = gpt2_model_ref.generate(query_tensors)
game_data["Targets"]= df_batch["target"]
game_data['response (after)'] = [gpt2_tokenizer.decode(response_tensors[i, :], skip_special_tokens = True) for i in range(bs)]

#### sentiment analysis of query/response pairs before/after
# texts = [q + r for q,r in zip(game_data['query'], game_data['response (before)'])]
# sentiment_inputs, attention_masks = build_bert_batch_from_txt(texts, sentiment_tokenizer, device)    
# rewards = sentiment_model.forward(sentiment_inputs, attention_masks)[0][:, 1].detach()
# game_data['rewards (before)'] = rewards.cpu().numpy()

# texts = [q + r for q,r in zip(game_data['query'], game_data['response (after)'])]
# sentiment_inputs, attention_masks = build_bert_batch_from_txt(texts, sentiment_tokenizer, device)    
# rewards = sentiment_model.forward(sentiment_inputs, attention_masks)[0][:, 1].detach()
# game_data['rewards (after)'] = rewards.cpu().numpy()
# DacHead
# store results in a dataframe
df_results = pd.DataFrame(game_data)


df_results.to_csv("result_final.csv")

