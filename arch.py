import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import RelaxedBernoulli

# import crcmod

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os
import argparse
import time
import math

class PositionalEncoding(nn.Module):
	"""
	https://pytorch.org/tutorials/beginner/transformer_tutorial.html
	"""

	def __init__(self, d_model, vocab_size=5000, dropout=0.1):
		super().__init__()
		self.dropout = nn.Dropout(p=dropout)

		pe = torch.zeros(vocab_size, d_model)
		position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(
			torch.arange(0, d_model, 2).float()
			* (-math.log(10000.0) / d_model)
		)
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer("pe", pe)

	def forward(self, x):
		x = x + self.pe[:, : x.size(1), :]
		return self.dropout(x)

class Identity(nn.Module):
    """A module that acts as an identity operation."""
    def forward(self, x):
        return x

class tr_enc(nn.Module):
	"""
	Text classifier based on a pytorch TransformerEncoder.
	"""

	def __init__(
		self,
		output_size,
		embeddings,
		nhead=8,
		dim_feedforward=2048,
		num_layers=6,
		dropout=0.1,
		activation="relu",
		classifier_dropout=0.1,
		use_pos = True
	):

		super().__init__()

		vocab_size, d_model = embeddings.size()
		assert d_model % nhead == 0, "nheads must divide evenly into d_model"

		self.emb = nn.Embedding.from_pretrained(embeddings, freeze=False)

		if use_pos:
			self.pos_encoder = PositionalEncoding(
				d_model=d_model,
				dropout=dropout,
				vocab_size=vocab_size,
			)
		else:
			self.pos_encoder = Identity()

		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=nhead,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			batch_first=True
		)
		self.transformer_encoder = nn.TransformerEncoder(
			encoder_layer,
			num_layers=num_layers,
		)
		self.classifier = nn.Linear(d_model, output_size)
		self.d_model = d_model

	def forward(self, x):
		# breakpoint()
		x = self.emb(x.clone()) * math.sqrt(self.d_model)
		# breakpoint()
		x = self.pos_encoder(x)
		x = self.transformer_encoder(x)
		x = x.mean(dim=1)
		x_logits = self.classifier(x)

		return x_logits


class MLP(nn.Module):
	def __init__(self, input_size=64, hidden_dim=640, output_size=64):
		super(MLP, self).__init__()
		hidden_size = input_size*hidden_dim
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, hidden_size)
		self.fc4 = nn.Linear(hidden_size, hidden_size)
		self.fc5 = nn.Linear(hidden_size, hidden_size)
		self.fc6 = nn.Linear(hidden_size, hidden_size)
		self.fc7 = nn.Linear(hidden_size, hidden_size)
		self.fc8 = nn.Linear(hidden_size, hidden_size)
		self.fc9 = nn.Linear(hidden_size, output_size)

		self.relu = nn.ReLU()

	def forward(self, x):#, mask = None):

		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)
		out = self.relu(out)
		out = self.fc3(out)
		out = self.relu(out)
		out = self.fc4(out)
		out = self.relu(out)
		out = self.fc5(out)
		out = self.relu(out)
		out = self.fc6(out)
		out = self.relu(out)
		out = self.fc7(out)
		out = self.relu(out)
		out = self.fc8(out)
		out = self.relu(out)

		logits = self.fc9(out)

		return logits


class Attention(nn.Module):
	def __init__(self, hidden_size, use_tanh, name = 'Bahdanau'):
		super(Attention, self).__init__()
		self.use_tanh = use_tanh
		self.name = name
		if name == 'Bahdanau':
			self.W_query = nn.Linear(hidden_size, hidden_size)
			self.W_ref = nn.Conv1d(hidden_size, hidden_size, 1, 1)
			self.V = nn.Parameter(torch.FloatTensor(hidden_size))
			self.V.data.uniform_(-(1./math.sqrt(hidden_size)), 1./math.sqrt(hidden_size))
			
	def forward(self, query, ref):
		# query: [batch_size, hidden_size]
		# ref: [batch_size, hidden_size, seq_len]
		if self.name == 'Bahdanau':
			seq_len = ref.size(2)
			expanded_query = self.W_query(query).unsqueeze(2).repeat(1,1,seq_len) # [batch_size x hidden_size x seq_len]
			ref = self.W_ref(ref) # [batch_size x hidden_size x seq_len]
			logits = torch.einsum('j, ijk -> ik', self.V, torch.tanh(expanded_query + ref))
			
		elif self.name == 'Dot':
			logits = torch.einsum('ij, ijk -> ik', query, ref)
			
		else:
			raise NotImplementedError
			
		if self.use_tanh:
			logits = 10*torch.tanh(logits)
		else:
			logits = logits
		return logits


class AttentionNet(nn.Module):
	def __init__(self, hidden_size):
		super(AttentionNet, self).__init__()
		self.context_attention = Attention(hidden_size, use_tanh=False)
		self.weight_attention = Attention(hidden_size, use_tanh=True)

	@staticmethod
	def apply_mask_to_logits(logits, mask):
		logits[mask.bool()] = -np.inf
		return logits

	def forward(self, query, ref, mask):
		# First step with glimpse attention
		context_logits = self.context_attention(query, ref)
		context_logits = self.apply_mask_to_logits(context_logits, mask)
		context_probs = F.softmax(context_logits, dim=1)
		context = torch.einsum('ijk,ik->ij', ref, context_probs)

		# Second step with pointer attention
		weight_logits = self.weight_attention(context, ref)
		
		return weight_logits  # Return logits to be softmaxed outside, if needed
