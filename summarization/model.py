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
from summarization.config import VietnewsConfig

def torch_load_all(dir):
    save_dict = {}
    for name in os.listdir(dir):
        if 'config' in name: continue
        save_dict[name.replace('.pt', '')] = torch.load(os.path.join(dir, name), map_location=torch.device('cpu'))

    return save_dict

class Bert_Embedding(nn.Module):
    def __init__(self, bert: AutoModel, config):
        super(Bert_Embedding, self).__init__()
        self.bert = bert
        self.bert_hidden = config.bert_hidden * config.bert_n_layers
        self.get_n_layers = config.bert_n_layers
        self.config = config
        
        self.windows_size = config.windows_size
        self.out_channels = config.out_channels
        self.lstm_embedding_size = len(self.windows_size) * config.MAX_SEQ_LEN  
        self.filters = nn.ModuleList([nn.Conv2d(1, self.out_channels,
                                                (i, self.bert_hidden)) for i in self.windows_size])
        self.relu = nn.ReLU()
        
    def forward(self, x, document_mask, attention_mask):
        lens = [mask_i.sum().item() for mask_i in document_mask]
        batch, doc_len, seq_len = list(x.shape)
        x = x.reshape((batch*doc_len, seq_len))
        attention_mask = attention_mask.reshape((batch*doc_len, seq_len))        
        last_hds, pooler_output, hidden_states = self.bert(x, attention_mask, output_hidden_states=True, return_dict=False)
        embeddings = torch.cat(hidden_states[-self.get_n_layers:], axis=-1)  # batch, doc_len, seq_len, self.bert_hidden
        # print(embeddings.shape)
        embeddings = embeddings.reshape((batch * doc_len, 1,  seq_len, self.bert_hidden))  # batch * doc_len, 1, MAX_SEQ_LEN, bert_hidden
        lstm_inputs = []
        for i in range(len(self.windows_size)):
            temp_out = self.filters[i](embeddings).squeeze(-1)  # batch * doc_len, self.out_channels, MAX_SEQ_LEN - self.windows_size[i] + 1
            cnn_result = torch.mean(temp_out, dim=1) # average along out_channels axis
            if cnn_result.shape[1] < self.config.MAX_SEQ_LEN: # pad cnn_result to MAX_SEQ_LEN
                pad_tensor = torch.zeros((cnn_result.shape[0], self.config.MAX_SEQ_LEN - cnn_result.shape[1])).to(cnn_result.device)
                cnn_result = torch.cat([cnn_result, pad_tensor], axis=1)
            lstm_inputs.append(cnn_result)
        lstm_inputs = torch.cat(lstm_inputs, dim=-1).reshape((batch, doc_len, self.lstm_embedding_size)) 
        lstm_inputs = lstm_inputs * torch.nn.functional.sigmoid(lstm_inputs)  # Swish 
        lstm_inputs = pack_padded_sequence(lstm_inputs, lens, batch_first=True, enforce_sorted=False)

        return lstm_inputs


class Document_Encoder(nn.Module):
    def __init__(self, embedding_size, config):
        super(Document_Encoder, self).__init__()

        self.config = config
        self.embedding_size = embedding_size
        self.doc_encoder = nn.LSTM(self.embedding_size, config.lstm_hidden, num_layers=1,
                            bidirectional=True, batch_first=True)

    def forward(self, lstm_inputs):
        _, doc_encoder_out = self.doc_encoder(lstm_inputs)

        return doc_encoder_out

class Sentence_Extractor(nn.Module):
    def __init__(self, embedding_size, config):
        super(Sentence_Extractor, self).__init__()

        self.config = config
        self.embedding_size = embedding_size
        self.sentence_extractor = nn.LSTM(self.embedding_size, config.lstm_hidden, num_layers=1,
                                  bidirectional=True, batch_first=True)
        self.dropout_layer = nn.Dropout(0.2)

    def forward(self, lstm_inputs, encoder_in):
        out_packed, (_, __) = self.sentence_extractor(lstm_inputs, encoder_in)
        out, out_lens = pad_packed_sequence(out_packed, batch_first=True)
        out = self.dropout_layer(out)
        return out

class Model(nn.Module):
    def __init__(self, bert, config):
        super(Model, self).__init__()
        self.config = config
        self.embeddings = Bert_Embedding(bert, config=config)
        self.doc_encoder = Document_Encoder(self.embeddings.lstm_embedding_size, config=config)
        self.sentence_extractor = Sentence_Extractor(self.embeddings.lstm_embedding_size, config=config)

        self.linear = Linear(config.lstm_hidden * 2, 1) 
        self.loss_func = nn.BCELoss()
        self.loss_padding_value = 0
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, document_mask, attention_mask, y=None):
        lstm_inputs = self.embeddings(x, document_mask, attention_mask)

        doc_encoder_out = self.doc_encoder(lstm_inputs)  
        encoder_in = doc_encoder_out

        out = self.sentence_extractor(lstm_inputs, encoder_in)
        out = self.sigmoid(self.linear(out).squeeze(-1))
        # print(out.shape)
        # out *= mask
        
        if y is not None:
            # print("First Y:", y)
            y = pad_sequence(y, batch_first=True, padding_value=self.loss_padding_value).type(torch.FloatTensor).to(out.device)
            # print("Out:", out, "Y:", y)
            loss = self.loss_func(out, y)
            # out = nn.functional.softmax(out, dim=-1)
            #print("Out:", out," Loss:",loss)
            return out, loss

        return out
    
    def predict(self, model, ids, document_mask, attention_mask):
      model.eval()
      with torch.no_grad():
        output_tensor = model(ids, document_mask, attention_mask)
      return output_tensor

class ESDataset(Dataset):
    def __init__(self, encodings, labels=None, keys=None):
        self.encodings = encodings
        self.labels = labels
        self.keys = keys
        self.encoding_keys = ['input_ids', 'attention_mask']

    def __getitem__(self, idx):
        item = {key: torch.tensor(self.encodings[idx][key]) for key in self.encoding_keys}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings)

def collate_fn(data):
    keys = data[0].keys()

    result = {k: [item[k] for item in data] for k in keys}
    input_ids = result['input_ids']
    result['document_mask'] = [torch.tensor([1] * len(input_ids[i])) for i in range(len(input_ids))]
    
    for k in result:
        if k != 'labels':
            result[k] = pad_sequence(result[k], batch_first=True)
    
    return result



    
