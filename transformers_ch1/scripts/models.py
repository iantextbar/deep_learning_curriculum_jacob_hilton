# dependencies
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.config import Configs

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# useful functions
def masking(att_patt):
  
  rows, cols = att_patt.shape[-2], att_patt.shape[-1] 

  temp = torch.ones((1, rows, cols))
  temp = torch.tril(temp)
  mask = (temp == 0).to(device)

  # masking upper triangle with -inf
  mask_mat = torch.masked_fill(att_patt, mask, -1e9)

  return mask_mat

def attention(Q, K, V, d_k, mask=True):

    den = np.sqrt(d_k)

    # attention function
    dot = torch.matmul(Q, K.transpose(-1, -2)) / den

    # masking if it's a decoder network
    if mask:
      dot = masking(dot)
    att_pat = F.softmax(dot, dim = -1)
    att = torch.matmul(att_pat, V)

    return att, att_pat


# models
class MultiHeadAttention(nn.Module):

  def __init__(self, config:Configs, mask=True):
    super(MultiHeadAttention, self).__init__()

    # hyperparams
    self.d_k = config.d_k
    self.d_model = config.d_model
    self.n_heads = config.n_heads
    self.batch = config.batch
    self.mask = mask

    self.Wq = nn.Linear(self.d_model, self.d_model)
    self.Wk = nn.Linear(self.d_model, self.d_model)
    self.Wv = nn.Linear(self.d_model, self.d_model)

    # last linear layer
    self.W0 = nn.Linear(self.d_model, self.d_model)


  def forward(self, pos_encode_toks):

    batch, seq_len, d_model = pos_encode_toks.shape
    Q = self.Wq(pos_encode_toks).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
    K = self.Wk(pos_encode_toks).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
    V = self.Wq(pos_encode_toks).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

    att, att_pat = attention(Q, K, V, d_k=self.d_k, mask=self.mask)

    # running last linear layer
    res = self.W0(att.transpose(1, 2).contiguous().view(batch, seq_len, -1))

    return res    
  
class MLP(nn.Module):

  def __init__(self, config:Configs):
    super(MLP, self).__init__()

    self.layer1 = nn.Linear(config.d_model, config.d_hid)
    self.layer2 = nn.Linear(config.d_hid, config.d_model)

  def forward(self, X):
    
    return self.layer2(F.relu(self.layer1(X)))

class LayerNorm(nn.Module):
  
  def __init__(self, config:Configs):
    super(LayerNorm, self).__init__()

    self.layernorm = nn.LayerNorm(config.d_model)

  def forward(self, X, sublayer_X):

    X = X + sublayer_X

    return self.layernorm(X)
  
class PositionalEncoding(nn.Module):
  
  def __init__(self, config:Configs):
    super(PositionalEncoding, self).__init__()
    
    self.d_model = config.d_model
    self.seq_len = config.seq_len

  def __op_pos_enc(self, dim, pos):

    return pos/(1e4**((dim)/self.d_model))

  def _pos_encoding_vec(self, pos):

    # for a given position, returns the vector of encodings
    pos_vec = np.zeros(self.d_model)
    for i in range(int(self.d_model)):
      if i % 2 == 0:
        pos_vec[i] = np.sin(self.__op_pos_enc(i, pos))
      else:
        pos_vec[i] = np.cos(self.__op_pos_enc(i-1, pos))

    return pos_vec

  def getPositionalEncoding(self):

    pos_encodings = np.zeros((self.seq_len, self.d_model))

    # gets the encoding vector for each position
    for pos in range(self.seq_len):
      for i in range(self.d_model):
        if i % 2 == 0:
          value = np.sin(pos/(1e4**(i/self.d_model)))
        else:
          value = np.cos(pos/(1e4**((i - 1)/self.d_model)))
        pos_encodings[pos, i] = value

    return torch.from_numpy(pos_encodings).type(torch.float32).to(device)
  
  def forward(self, embeddings):

    pos_enc = self.getPositionalEncoding()

    return embeddings + pos_enc

 
class DecoderLayer(nn.Module):
  
  def __init__(self, config:Configs, mask=True):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(config, mask=mask)
    self.dropout1 = nn.Dropout(config.dropout)
    self.layernorm1 = LayerNorm(config)
    self.ff = MLP(config)
    self.dropout2 = nn.Dropout(config.dropout)
    self.layernorm2 = LayerNorm(config)

  def forward(self, X):
    
    # first attention block
    X = self.layernorm1(X, self.dropout1(self.mha1(X)))
  
    # feed forward network
    X = self.layernorm2(X, self.dropout2(self.ff(X)))
  
    return X

class Transformer(nn.Module):

  def __init__(self, vocab_size, config:Configs, mask=True):
    super(Transformer, self).__init__()

    # initial hyperparams
    self.vocab_size = vocab_size
    self.batch = config.batch
    self.d_model = config.d_model
    self.n_layers = config.n_layers
    self.d_k = config.d_k

    # embedding layer
    self.embedding = nn.Embedding(vocab_size, self.d_model)

    # positional encodding
    self.pos_enc = PositionalEncoding(config)

    # dropout
    self.dropout = nn.Dropout(config.dropout)

    # decoder blocks
    self.blocks = nn.ModuleList([DecoderLayer(config, mask=mask) for _ in range(self.n_layers)])

    # unembedding
    self.unembed = nn.Linear(self.d_model, self.vocab_size)

  def forward(self, X):
    
    embeddings = self.embedding(X)
    out = self.dropout(self.pos_enc(embeddings))

    for i, l in enumerate(self.blocks):
      
      out = l(out)

    logits = self.unembed(out)

    return logits
