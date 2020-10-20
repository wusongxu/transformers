import torch
import torch.nn as nn
import torch.nn.functional as functional

class DSSM(nn.Module):
    def __init__(self,config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.word_num,config.hidden_size)
        self.rnn = nn.LSTM(config.hidden_size,config.hidden_size,num_layers=1,bidirectional=True,batch_first=True)
        self.Dense = nn.Linear(config.hidden_size*2,config.)