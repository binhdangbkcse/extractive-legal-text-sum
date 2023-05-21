from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from summarization.model import *
from django.views.decorators.csrf import csrf_exempt

import json
import numpy as np
import pandas as pd
import re
import os

from sklearn.utils import shuffle

from transformers import AutoModel, AutoTokenizer

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import LSTM, Conv2d, Linear
from torch.nn.functional import max_pool2d
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from underthesea import sent_tokenize

import PyPDF2
import sys
sys.path.append("summarization_model/summarization/config.py")
sys.path.append("summarization_model/summarization/model.py")
print(sys.path)
from summarization.model import *

from summarization.config import VietnewsConfig
# Create your views here.

def word_compound(text):
  
  with open("../phobert-base/tokenizer.json") as vocab: 
    vocab = json.load(vocab)["model"]["vocab"]
    keys = list(vocab.keys())
    new_keys =[]
    for key in keys:
        new_keys += [key.lower()]
    keys = new_keys
    keys = list(filter(lambda key : key if re.search(".+\_.+", key) else None, keys))
    
  check = lambda x, key: True if x in key else False
  text = text.lower()
  text = text.replace(", ", " , ")
  text = text.replace("./.", " .")
  text = text.replace(". ", " . ")
  text = text.replace(": ", " : ")
  text = text.replace("; ", " ; ")
  text = re.sub(".$", " .", text)
  word_list = text.split(" ")
  temp = word_list[0]
  checked = False
  return_string = ""
  for i in range(1, len(word_list)):
    new_word = temp + "_" + word_list[i]
    for key in keys:
      if check(new_word, key):
        temp = new_word
        checked = True
        break
    if checked:
      checked = False
      continue
    else:
      return_string += temp + " "
      temp = word_list[i]
  if return_string != "":
    if return_string[-1] == " ":
      return_string = return_string[:-1:]
  return return_string
  
def text_normalize(text):
    text = re.sub(";.", ".", text)
    text = re.sub("^[+-]", "", text)
    text = re.sub(". [+-] ", ". ", text)
    text = text.lower()
    text = text.replace("\n"," ")
    text = text.strip()
    if text[-1] != ".":
      text += "."
    text = text.replace(";.", ".")
    text = text.replace(" _ . ", "_")
    text = word_compound(text)
    return text

def get_doc_content(file):
  read_file = PyPDF2.PdfReader(file) 
  full_text = ""
  for index in range(len(read_file.pages)):
    content_page = read_file.pages[index].extract_text()
    content_page = content_page.replace(";\n", ". ")
    content_page = content_page.replace("\n", " ")
    content_page = re.sub(" +",
                              " ", content_page)
    content_page = re.sub("( /)|(/ )",
                          "/", content_page)
    content_page = re.sub("( -)|(- )",
                          "-", content_page)
    content_page = re.sub("(QH)\s([0-9]+)",
                          '\\1\\2', content_page)
    content_page = text_normalize(content_page)
    # if index == 0:
    #   word_list = content_page.split()
    #   for i, word in enumerate(word_list):
    #     if word in ["quyết_định", "kế_hoạch"]:
    #       word_list = word_list[i:]
    #       content_page = ' '.join(word_list)
    full_text += content_page
    if ("./." in content_page) or ("nơi nhận:" in content_page):
      break
  return full_text

def text_separate(full_text):
  sample_list = []
  full_text = sent_tokenize(full_text)
  num_of_sent = len(full_text)
  if num_of_sent < 8:
      sample_list += [' '.join(str(text) for text in full_text)]
  elif num_of_sent < 10:
      half_sent = num_of_sent//2
      sample_list += [' '.join(str(text) for text in full_text[:half_sent]), ' '.join(str(text) for text in full_text[half_sent:num_of_sent])]
  else:
      while len(full_text) >= 10 :
        sample_list += [' '.join(str(text) for text in full_text[:10])]
        full_text = full_text[5:]
      if num_of_sent < 8:
          sample_list += [' '.join(str(text) for text in full_text)]
      elif num_of_sent < 10:
          half_sent = num_of_sent//2
          sample_list += [' '.join(str(text) for text in full_text[:half_sent]), ' '.join(str(text) for text in full_text[half_sent:num_of_sent])]

  return remove_short_sentence(sample_list)

def remove_short_sentence(samples):
  for index in range(len(samples)):
    sentence_list = sent_tokenize(samples[index])
    for i in range(len(sentence_list)):
      if i >= len(sentence_list):
        break
      if len(sentence_list[i])==0 or len(sentence_list[i])==1 or len(sentence_list[i])==2:
        sentence_list.pop(i)
        continue
      if len(sentence_list[i].split(" ")) <= 5:
        if i+1 < len(sentence_list):
          new_sentence = sentence_list[i][:-1:]+" "+sentence_list[i+1]
          sentence_list[i] = new_sentence
          sentence_list.pop(i+1)
        else:
          new_sentence = sentence_list[i-1][:-1:]+" "+sentence_list[i]
          sentence_list[i-1] = new_sentence
          sentence_list.pop(i)
    samples[index] = " ".join(sentence_list)
  return samples


def docs_preprocess (file):
  doc_content = get_doc_content(file) #trường hợp file đã đọc thành text thì truyền text vào đây
  docs_raw = text_separate(doc_content)
  docs_raw = [sent_tokenize(doc) for doc in docs_raw]
  docs = docs_raw
  return docs_raw, docs

def get_summary(tokenizer, model, docs, config): 
  encodings = []
  for doc in docs:
    encodings.append(tokenizer(doc[:config.MAX_DOC_LEN], truncation=True,
                                max_length=config.MAX_SEQ_LEN, padding='max_length'))
  predict_dataset = ESDataset(encodings)
  predict_loader = DataLoader(predict_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
  probs = []
  for item in predict_loader:
    ids = item['input_ids'].to(config.device)
    # print(ids.shape)
    document_mask = item['document_mask'].to(config.device)
    attention_mask = item['attention_mask'].to(config.device)
    # print(ids)
    prob = model.predict(model,ids, document_mask, attention_mask).tolist()
    probs += [prob]
  max_prob_sents = [np.argmax(prob) for prob in probs]
  return max_prob_sents

def post_process(result):
  new_result = []
  for r in result:
    r = re.sub(" +", " ", r)
    r = re.sub(" +_", " ", r)
    r = r.replace("_", " ")
    r = r.replace(" ,", ",")
    r = r.replace(" .", ".")
    r = r.replace(" :", ":")
    r = r.replace(" ;", ";")
    new_result += [r]
  return new_result


@csrf_exempt
def summarization(req):
    if req.method == "POST":
        file = req.FILES.get("document")
        final_dict = torch_load_all('best-model') #tải model của mình về r gắn đường dẫn vào đây
        with open('best-model/config.json', 'r') as f:
          data = f.read()
          config = json.loads(data)
          config = json.loads(config)
          config = VietnewsConfig(config['train_data_dir'], config['val_data_dir'], config['test_data_dir']
                                  , config['bert_name'], config['MAX_SEQ_LEN'], config['MAX_DOC_LEN'], config['bert_hidden'],
                                  config['bert_n_layers'], config['windows_size'], config['out_channels'],
                                  config['lstm_hidden'], config['device'], config['batch_size'],
                                  config['num_epochs'], config['warmup_steps'], config['gradient_accumulation_steps'],
                                  config['print_freq'], config['save_dir'])
        config.bert_name = "phobert-base"
        config.device = 'cpu'
        bert = AutoModel.from_pretrained(config.bert_name)
        tokenizer = AutoTokenizer.from_pretrained(config.bert_name)
        model = Model(bert, config).to(config.device)
        model.load_state_dict(final_dict['model_state_dict'])
        docs_raw, docs = docs_preprocess(file)
        sent_ids = get_summary(tokenizer, model, docs,config)
        result = [docs_raw[i][sent_ids[i]] for i in range(len(docs_raw))]
        result = post_process(result)
        return JsonResponse(result, safe=False)
    return HttpResponse(status=500)