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
import os, time

torch.autograd.set_detect_anomaly(False)


from tqdm import tqdm

import logging


from arch import MLP, tr_enc, AttentionNet

from polar_ber import polar_ber

from compute_snr_schedule import rs_5g

# random seed
seed = 42

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Target code params
Nt = 32

Ki = Nt

target_bler = 0.01

decoder_type = "scl_crc" # "sc_naive"/"scl_crc"

channel = "awgn"	# "awgn"/"rayleigh"
min_err = 100

if decoder_type == "scl_crc":
	crc_len = 4
	list_size = 8
else:
	crc_len = 0
	list_size = 1

train_model = 1
start_epoch = 0

K_list = np.arange(1+crc_len, Nt+1)

## compute SNR required for target BLER using 3GPP sequence; default target BLER is 0.01

if target_bler == 0.01:
	if Nt == 32:
		if (decoder_type == "sc_naive"):
			SNR_schedule = np.array([-10.76, -6.79, -5.76, -5.06, -4.29, -3.32, -2.62, -2.07, -1.57, -1.22, -0.81, -0.43, -0.21, 0.13, 0.49, 0.92, 1.17, 1.63, 1.95, 2.34, 2.57, 2.74, 3.05, 3.28, 3.5, 3.87, 4.39, 4.93, 5.38, 5.77, 6.12, 7.66])
  
		elif (decoder_type == "scl_crc"):
			SNR_schedule = np.array([-7.74, -6.42, -5.69, -5.12, -4.46, -3.33, -2.78, -2.41, -2.21, -1.85, -1.5, -1.16, -0.82, -0.31, -0.07, 0.29, 0.74, 1.68, 1.67, 1.89, 2.15, 2.34, 3.01, 3.1, 3.23, 3.58, 4.35, 8.0]) # FIX ME

	elif Nt == 64:
		if (decoder_type == "sc_naive"):
				if channel == "awgn":
					SNR_schedule = [-13.79, -9.8, -8.79, -8.1, -7.31, -6.29, -5.34, -5.06, -4.76, -4.39, -4.1, -3.8, -3.4, -3.16, -2.89, -2.52, -2.27, -2.06, -1.78, -1.54, -1.44, -1.05, -0.93, -0.46, -0.4, -0.26, -0.16,
                     					-0.02, 0.19, 0.31, 0.45, 0.81, 0.99, 1.0, 1.11, 1.24, 1.39, 1.59, 1.78, 1.97, 2.02, 2.24, 2.39, 2.64, 2.78, 2.96, 3.1, 3.25, 3.38, 3.59, 3.73, 3.96, 4.03, 4.28, 4.41, 4.62, 4.86, 5.1, 5.45, 5.78, 6.13, 6.48, 6.75, 8.12]
				elif channel == "rayleigh":
					SNR_schedule = [-12.6, -8.48, -7.46, -6.62, -5.77, -4.68, -3.55, -3.16, -2.82, -2.39, -2.05, -1.73, -1.25, -0.93, -0.57, -0.03, 0.18, 0.62, 1.01, 1.36, 1.54, 2.09, 2.24, 2.94, 3.07, 3.3, 3.44, 3.62, 4.04, 4.27, 4.48, 5.07, 5.4, 5.45, 5.57, 
                     					5.8, 6.11, 6.56, 7.01, 7.35, 7.5, 7.81, 8.21, 8.85, 9.12, 9.54, 9.86, 10.25, 10.64, 11.3, 11.51, 12.14, 12.48, 13.15, 13.56, 14.3, 15.19, 15.91, 17.21, 18.71, 20.09, 21.75, 23.23, 32.02]
		if (decoder_type == "scl_crc"):
			if crc_len == 11:	
				if target_bler == 0.01 and list_size == 4:
					SNR_schedule = np.array([-10.08, -8.86, -8.05, -7.65, -6.84, -6.65, -5.84, -5.33, -5.05, -4.71, -4.35, -4.12, -3.83, -3.57, -3.32, -3.0, -2.93, -2.64, -2.49, -2.2, -2.06, -1.84, -1.72, -1.3, -1.21, -0.92, -0.76, -0.53, -0.44, -0.21, -0.05, 0.06, 0.33, 0.47, 
									0.55, 0.71, 0.9, 0.91, 1.2, 1.33, 1.49, 1.67, 2.12, 2.29, 2.35, 2.57, 2.8, 2.92, 3.11, 3.33, 3.45, 3.69, 3.88, 4.08, 4.32, 4.47, 4.84, 5.33, 5.75, 7.25])
				elif target_bler == 0.01 and list_size == 8:
					SNR_schedule = SNR_schedule = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, -6.925, -6.03, -5.59, -5.09, -4.715, -4.345, -4.12, -3.825, -3.69, -3.46, -3.07, -2.985, -2.765, -2.52, -2.36, -2.08, -1.935, -1.64, -1.375, -1.155, -0.995, -0.96, -0.595, -0.45, 
                                                                                                                  -0.36, -0.1, 0.045, 0.185, 0.485, 0.435, 0.65, 0.705, 1.025, 1.06, 1.19, 1.39, 1.75, 1.91, 2.055, 2.265, 2.48, 2.66, 2.83, 3.08, 3.33, 3.51, 3.67, 3.86, 4.185, 4.42, 4.735, 5.04, 6.5]
			elif crc_len == 4:
				if channel == "awgn":
					SNR_schedule = [100.0, 100.0, 100.0, 100.0, -10.38, -9.48, -8.435, -8.04, -6.92, -6.93, -6.155, -5.655, -5.29, -4.67, -4.68, -4.3, -4.095, -3.88, -3.505, -3.32, -3.19, -2.95, -2.82, -2.525, -2.355, -2.035, -1.88, -1.595, -1.3, -1.15, -0.96, -0.745, -0.6, -0.405, -0.245, -0.14, 0.105, 0.195, 
						0.37, 0.52, 0.62, 0.785, 0.97, 1.075, 1.29, 1.46, 2.05, 2.335, 2.285, 2.37, 2.61, 2.83, 3.015, 3.075, 3.36, 3.645, 3.64, 4.055, 4.135, 4.29, 4.57, 5.23, 5.575, 6.5]
	
				elif channel == "rayleigh":
					SNR_schedule = [100.0, 100.0, 100.0, 100.0, -9.06, -8.19, -7.085, -6.625, -6.07, -5.345, -4.76, -3.99, -3.74, -3.2, -3.035, -2.66, -2.36, -1.875, -1.51, -1.28, -0.995, -0.825, -0.4, -0.18, 0.17, 0.42, 0.68, 1.155, 1.53, 1.93, 2.11, 2.675, 2.995, 2.945, 3.14, 3.475, 3.705, 4.195, 4.24, 4.435, 4.625, 
									4.91, 5.38, 5.44, 5.81, 6.12, 6.51, 7.04, 7.74, 8.25, 8.4, 8.795, 9.1, 9.7, 10.305, 10.685, 11.255, 12.07, 12.935, 13.05, 13.975, 15.975, 17.225, 20.0]
	elif Nt == 128:
		if (decoder_type == "sc_naive"):
			SNR_schedule = [-16.8, -12.83, -11.86, -11.09, -10.24, -9.4, -8.3, -8.07, -7.07, -7.17, -6.86, -6.74, -6.48, -6.33, -6.07, -5.82, -5.49, -5.33, -5.07, -4.79, -4.44, -4.34, -4.23, -4.02, -3.96, -3.73, -3.65, -3.45, -3.32, -3.11, -3.1, -2.9, -2.77, -2.52, -2.4,
					-2.31, -2.28, -2.19, -1.8, -1.75, -1.69, -1.6, -1.57, -1.47, -1.37, -1.36, -1.22, -1.08, -0.96, -0.93, -0.78, -0.75, -0.66, -0.56, -0.47, -0.33, -0.3, -0.15, -0.08, 0.01, 0.15, 0.18, 0.27, 0.38, 0.46, 0.54, 0.62, 0.72, 0.78, 0.92, 0.95, 0.95, 1.15, 1.26,
					1.46, 1.54, 1.59, 1.63, 1.7, 1.74, 1.77, 1.84, 1.89, 1.97, 2.06, 2.15, 2.18, 2.32, 2.41, 2.59, 2.62, 2.69, 2.81, 2.84, 2.92, 2.95, 3.11, 3.2, 3.32, 3.44, 3.5, 3.57, 3.67, 3.75, 3.87, 3.96, 4.07, 4.2, 4.24, 4.35, 4.43, 4.51, 4.67, 4.75, 4.91, 5.01, 5.18,
					5.29, 5.49, 5.58, 5.75, 5.98, 6.27, 6.47, 6.76, 7.02, 7.27, 8.48]
		if (decoder_type == "sc_sys"):
			SNR_schedule = [-16.79, -12.84, -11.85, -11.11, -10.25, -9.4, -8.26, -8.05, -7.1, -7.04, -6.84, -6.72, -6.54, -6.32, -6.1, -5.88, -5.57, -5.34, -5.08, -4.78, -4.49, -4.33, -4.19, -3.94, -3.87, -3.77, -3.59, -3.46, -3.34, -3.13, -3.04, -2.86, -2.82, -2.43, -2.41,
					-2.34, -2.26, -2.18, -1.83, -1.73, -1.65, -1.56, -1.52, -1.5, -1.37, -1.34, -1.28, -1.04, -1.0, -0.89, -0.89, -0.77, -0.63, -0.53, -0.45, -0.35, -0.25, -0.16, -0.06, -0.05, 0.17, 0.25, 0.25, 0.38, 0.49, 0.57, 0.61, 0.68, 0.83, 0.92, 0.98, 0.96, 1.19, 1.24,
					1.48, 1.54, 1.56, 1.54, 1.73, 1.77, 1.76, 1.85, 1.88, 1.99, 2.01, 2.18, 2.19, 2.37, 2.44, 2.52, 2.69, 2.67, 2.81, 2.86, 2.91, 3.0, 3.11, 3.21, 3.35, 3.41, 3.45, 3.55, 3.67, 3.76, 3.86, 3.95, 4.07, 4.18, 4.27, 4.38, 4.37, 4.5, 4.66, 4.81, 4.9, 4.99, 5.16,
					5.29, 5.51, 5.62, 5.75, 5.95, 6.24, 6.51, 6.79, 7.05, 7.24, 8.54]
   
		if decoder_type == "scl_crc":
			if crc_len == 4:
				if channel == "awgn":
					SNR_schedule = [-13.53, -12.28, -11.56, -11.0, -10.45, -10.08, -9.42, -9.24, -8.52, -8.21, -7.85, -7.61, -7.37, -7.11, -6.83, -6.58, -6.41, -6.22, -6.03, -5.88, -5.61, -5.57, -5.35, -5.24, -5.0, -4.75, -4.72, -4.53, -4.36, -4.26, -4.1, -3.96, -3.84, -3.62,
						-3.63, -3.43, -3.32, -3.19, -3.08, -2.99, -2.85, -2.76, -2.61, -2.5, -2.43, -2.29, -2.18, -2.11, -1.99, -1.93, -1.84, -1.73, -1.62, -1.53, -1.44, -1.34, -1.25, -1.14, -1.02, -0.94, -0.82, -0.72, -0.67, -0.55, -0.47, -0.35, -0.26, -0.2, -0.13, -0.03,
						0.08, 0.14, 0.23, 0.31, 0.39, 0.47, 0.58, 0.65, 0.77, 0.83, 0.9, 1.03, 1.1, 1.25, 1.28, 1.37, 1.44, 1.48, 1.66, 1.76, 1.81, 1.87, 2.01, 2.08, 2.25, 2.31, 2.39, 2.41, 2.55, 2.61, 2.77, 2.85, 3.03, 3.12, 3.3, 3.44, 3.46, 3.56, 3.67, 3.81, 3.95, 4.05,
						4.17, 4.33, 4.42, 4.57, 4.73, 4.91, 5.08, 5.28, 5.61, 5.96, 6.34, 8.0]
				elif channel == "rayleigh":
					SNR_schedule = [-12.32, -11.04, -10.32, -9.76, -9.2, -8.76, -8.09, -7.89, -7.11, -6.76, -6.43, -6.12, -5.92, -5.61, -5.31, -5.01, -4.81, -4.57, -4.38, -4.22, -3.94, -3.86, -3.58, -3.44, -3.21, -2.93, -2.82, -2.63, -2.4, -2.26, -2.05, -1.89, -1.75, -1.53,
                     -1.46, -1.28, -1.14, -0.94, -0.82, -0.68, -0.51, -0.37, -0.19, -0.03, 0.09, 0.24, 0.37, 0.55, 0.76, 0.79, 0.9, 1.01, 1.17, 1.33, 1.44, 1.58, 1.73, 1.85, 1.99, 2.22, 2.32, 2.49, 2.65, 2.8, 2.93, 3.16, 3.17, 3.32, 3.47, 3.57, 3.78, 3.86, 4.01, 4.15, 4.34,
                     4.41, 4.61, 4.77, 4.97, 5.02, 5.19, 5.33, 5.57, 5.83, 5.94, 6.02, 6.16, 6.31, 6.64, 6.76, 6.98, 7.17, 7.37, 7.55, 7.86, 7.95, 8.2, 8.19, 8.47, 8.73, 9.06, 9.23, 9.63, 9.92, 10.37, 10.6, 10.63, 10.97, 11.37, 11.7, 11.94, 12.39, 12.76, 13.19, 13.48, 13.97,
                     14.57, 15.05, 15.7, 16.33, 17.79, 19.11, 20.8, 22.0]

# select the last len(K_list) elements from the SNR_schedule
SNR_schedule = SNR_schedule[-len(K_list):]

# find the postion in K_list where Kc = Nt//2
for iK in range(len(K_list)):
	if K_list[iK] == Nt//2:
		break
K_ref = K_list[iK]
SNR_ref = SNR_schedule[iK]

# select the number/type of learnable parameters
network_mode = "setnet" # sl/ff/"sig"

# arch = "tr" # "mlp"/"tr"
arch = "attn"
# transformer config
nhead = 4
dim_feedforward = 256
num_layers = 2

# mlp config
# actual hidden dim = hidden_dim * Nt
depth = 9

embedding_size = Nt
hidden_dim = int(embedding_size*10)
output_size = Nt
 
if arch == "mlp":
	batch_size  = int(1e2)
else:
	batch_size = int(1e2)

num_epochs = int(4e4)

learning_rate = 0.0002

# SNRs = np.array([1.0])

def seed_everything(seed):
	os.environ["PYTHONHASHSEED"] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True

class SetNet(nn.Module):
	def __init__(self, input_size=64, output_size=64, embedding_size=128, arch="attn"):
		super(SetNet, self).__init__()

		self.arch = arch
		self.input_size = input_size
		self.output_size = output_size

		if (arch == "mlp"):
			self.mlp = MLP(input_size=embedding_size, hidden_dim=hidden_dim, output_size=output_size)
			self.embedding = nn.Embedding(input_size, embedding_size)
		elif arch == "attn":
			self.attention = AttentionNet(embedding_size)
			# Last embedding is for zero input set.
			self.embedding = nn.Embedding(input_size+1, embedding_size)
		else:
			self.enc = tr_enc(output_size, self.embedding.weight, nhead=nhead, dim_feedforward=dim_feedforward, num_layers=num_layers, dropout=0.1, activation="relu", classifier_dropout=0.1)
			self.embedding = nn.Embedding(input_size, embedding_size)
		self.softmax = nn.Softmax(dim=1)
		self.embedding_size = embedding_size

	@staticmethod
	def apply_mask_to_logits(logits, mask):
		logits[mask.bool()] = -np.inf
		return logits

	@staticmethod
	def update_mask(mask, idxs):
		clone_mask = mask.clone()
		if idxs is not None:
			clone_mask[[i for i in range(mask.size(0))], idxs.detach()] = 1

		return clone_mask
	
	def forward(self, in_set, batch_size, strategy='sampling'):
		
		start_len = len(in_set)
  
		seq_len = Nt - start_len
  
		if (self.arch == "mlp"):
			# create a permutation invariant embedding for the input set
			emb_init = torch.zeros(Nt).to(device)

			for ix in in_set:
				emb_init += self.embedding(ix)
	
			# replicate the embedding to create a batch of size batch_size
			emb = emb_init.repeat(batch_size, 1)
		elif self.arch == 'attn':
			emb = self.embedding(torch.tensor([self.input_size], dtype=torch.long, device = device))
			if len(in_set) > 0 :
				emb = emb + torch.sum(self.embedding(in_set), dim=0)
			ref_embeddings = self.embedding(torch.arange(self.input_size, device = next(self.embedding.parameters()).device)).unsqueeze(0).repeat(batch_size, 1, 1)

		probs_list = []
		idxs_list = torch.zeros(batch_size, Nt, dtype=torch.int).to(device)
		idxs_list[:,:start_len] = in_set.detach()
  
		idxs = in_set.detach()
		mask = torch.zeros(batch_size, Nt).byte()
		num_init = len(in_set)
		for ii in range(num_init):
			idxs_curr = torch.tensor([idxs[ii], ]).to(device)
			mask = self.update_mask(mask, idxs_curr)

		for ik in range(seq_len):			
			# predict the next index
			if (self.arch == "mlp"):
				logits = self.mlp(emb)
			elif self.arch == 'attn':
				logits = self.attention(emb, ref_embeddings, mask)
			else:
				logits = self.enc(idxs_list)

			logits = self.apply_mask_to_logits(logits, mask)

			probs = F.softmax(logits, dim=1)
			
			if strategy=='sampling':
				idxs=probs.multinomial(1).squeeze(1)
			elif strategy=='greedy':
				idxs=probs.argmax(1)
			idxs_list[:,ik+start_len] = idxs.detach()
			mask = self.update_mask(mask, idxs)
   
			prob_i = probs[[i for i in range(batch_size)], idxs.detach()]
			probs_list.append(prob_i)

			if (self.arch in ["mlp", "attn"]):
				# add the embedding of the new index to the existing embedding
				emb = emb.clone() + self.embedding(idxs)
			

		return probs_list, idxs_list



if __name__ == '__main__':
	# set seed for the run
	seed_everything(seed)
 
	sim_str = f'N_{Nt}_{decoder_type}_{target_bler}'
 
	rs_5g = rs_5g[rs_5g<Nt]
	rs_5g = rs_5g[::-1]

	if decoder_type == "scl_crc":
		sim_str = f'N_{Nt}_{list_size}_{decoder_type}_{crc_len}_{target_bler}'
 
	parent_folder = "results"
	tr_str = f"arch_{arch}_lr_{learning_rate}"
 
	log_dir = f"{parent_folder}/logs/{tr_str}/{sim_str}"
	bler_lut_dir = f"{parent_folder}/bler_luts/{tr_str}/{sim_str}"
	model_dir =  f"{parent_folder}/models/{tr_str}/{sim_str}"
 
	os.makedirs(model_dir, exist_ok=True)
	os.makedirs(log_dir, exist_ok=True)
	os.makedirs(bler_lut_dir, exist_ok=True)

	# load the bler_lut if it exists
	if os.path.exists(f"{bler_lut_dir}/bler_{sim_str}.pkl"):
		try:
			with open(f"{bler_lut_dir}/bler_{sim_str}.pkl", "rb") as f:
				print(f"Loading bler_lut from {bler_lut_dir}/bler_{sim_str}.pkl")
				bler_lut = pickle.load(f)
			with open(f"{bler_lut_dir}/ber_{sim_str}.pkl", "rb") as f:
				print(f"Loading ber_lut from {bler_lut_dir}/bler_{sim_str}.pkl")
				ber_lut = pickle.load(f)
		except:
			print("Error loading LUTs")
	# breakpoint()
	log_file_name = f"{log_dir}/{sim_str}_{network_mode}_nheads_{nhead}_hd_{dim_feedforward}_num_layers_{num_layers}_Nt_{Nt}_Ki_{Ki}_lr_{learning_rate}.txt"
	logging.basicConfig(format='%(message)s', filename=log_file_name, encoding='utf-8', level=logging.INFO)
 
	logging.info(f"Network mode: {network_mode}")
	logging.info(f"Network arch: {arch}")
	logging.info(f"Target BLER: {target_bler}")
	logging.info(f"SNR schedule: {SNR_schedule}")
	logging.info(f"K list: {K_list}")
	logging.info(f"Number of epochs: {num_epochs}")
	logging.info(f"Learning rate: {learning_rate}")
	logging.info(f"Batch size: {batch_size}")
	logging.info(f"Number of heads: {nhead}")
	logging.info(f"Hidden dim: {dim_feedforward}")
	logging.info(f"Number of layers: {num_layers}")
	logging.info(f"Number of bits: {Nt}")
	logging.info(f"Number of info bits: {Ki}")
	logging.info(f"Number of learnable parameters: {hidden_dim}")
	# logging.info(f"SNRs: {SNRs}")
	logging.info(f"Seed: {seed}")
	logging.info("#############################################")
	try:
		# create the setnet model
		if (network_mode == "setnet"):
			setnet = SetNet(input_size=Nt, output_size=Nt, embedding_size = embedding_size, arch = arch).to(device)
	
			optimizer = optim.Adam(setnet.parameters(), lr = learning_rate)

			beta_decay = 0.99

			min_loss = 1e10
			min_seq = 0

			# create a look up table to store bler values corresponding to each rate, if variables bler_lut and ber_lut do not exist
			if 'bler_lut' not in globals():
				bler_lut = {}
			if 'ber_lut' not in globals():
				ber_lut = {}

			if train_model:
				# load any existing model
				if start_epoch > 0:
					print(f"Loading pretrained model from {model_dir}/setnet_{sim_str}_{start_epoch}.pt")
					logging.info(f"Loading pretrained model from {model_dir}/setnet_{sim_str}_{start_epoch}.pt")
					setnet.load_state_dict(torch.load(f"{model_dir}/setnet_{sim_str}_{start_epoch}.pt"))

				for epoch in tqdm(range(start_epoch,num_epochs+1)):

					# set the model to train mode
					setnet.train()
				
					# use existing frozen set to generate next information position
					# in_data = torch.tensor([Nt-1, ]).to(device)
     
					in_data = torch.tensor(rs_5g[:crc_len].copy()).to(device)
					# in_data = torch.tensor([ ]).to(device)

					# sample a sequence of indices
					probs_list, seq_list = setnet(in_data, batch_size, strategy='sampling')

					logprobs = 0
					for prob in probs_list:
						logprob = torch.log(prob)
						logprobs=logprobs+logprob
					logprobs[logprobs<-1000]=0.

					# compute the reward for target_rate using py_aff3ct
					R = torch.zeros(batch_size).to(device)
					mean_ber = 0
					mean_bler = 0
					mean_bler_ref = 0
					seq_list = seq_list.detach().cpu().numpy()
     
					for i_t in range(batch_size):
         
						cur_seq = seq_list[i_t]
      
						for iK in range(len(K_list)):
							Kc = K_list[iK]
							SNRs = np.array([SNR_schedule[iK], ])
	
							info_positions = cur_seq[:Kc]
							# get the frozen_pos vector
							frozen_bits = np.ones(Nt, dtype=bool)
							frozen_bits[info_positions] = 0
			
							# compute permutation invariant representation of info_positions
							info_pos_pi = frozenset(info_positions)

							# check if this key exists in bler_lut dictionary
							if info_pos_pi in bler_lut:
								ber = ber_lut[info_pos_pi]
								bler = bler_lut[info_pos_pi]
							else:

								ber, bler = polar_ber(Nt, Kc, channel, min_err, frozen_bits, SNRs, decoder_type, list_size, crc_len)

								ber_lut[info_pos_pi] = ber
								bler_lut[info_pos_pi] = bler

							R[i_t] += bler
				
							mean_ber += ber
							mean_bler += bler
			
							# store bler corresponding to rate 1/2
							if (Kc == Nt//2):
								bler_ref = bler
								mean_bler_ref += bler
						# breakpoint()
					mean_ber /= (batch_size*len(K_list))
					mean_bler /= (batch_size*len(K_list))
					mean_bler_ref /= batch_size
		
					if epoch==start_epoch:
						R_avg=R
					else:
						R_avg = (R_avg*beta_decay)+((1.-beta_decay)*R)

					advantage = (R-R_avg).detach()
					loss = (advantage*logprobs).mean()
		
					optimizer.zero_grad()
					loss.backward()
					torch.nn.utils.clip_grad_norm_(setnet.parameters(), 2., norm_type=2)
					optimizer.step()

					if epoch % 100 == 0:
						print(f"Epoch: {epoch}, Loss: {loss.item()}, Mean BLER = {mean_bler}, Mean BLER at rate 1/2 = {mean_bler_ref}, Recent I = {info_positions}")
						logging.info(f"Epoch: {epoch}, Loss: {loss.item()}, Mean BLER = {mean_bler}, Mean BLER at rate 1/2 = {mean_bler_ref}, Recent I = {info_positions}")
			
						with torch.no_grad():
							# test the model
							in_data = torch.tensor(rs_5g[:crc_len].copy()).to(device)

							# sample a sequence of indices
							probs_list, seq_list = setnet(in_data, 1, strategy='greedy')
					
							final_seq = seq_list[0]

							# compute BLER using py_aff3ct
							info_positions = final_seq[:K_ref].detach().cpu().numpy()
       
							# get the frozen_pos vector
							frozen_bits = np.ones(Nt, dtype=bool)
							frozen_bits[info_positions] = 0

							SNRs = np.array([ SNR_ref, ])

							ber, bler = polar_ber(Nt, K_ref, channel, min_err, frozen_bits, SNRs, decoder_type, list_size, crc_len)
							# print final BLER
							print(f"Final rate 1/2 BLER: {bler} at SNR = {SNRs[0]} with I = {info_positions}")
							logging.info(f"Final rate 1/2 BLER: {bler} at SNR = {SNRs[0]} with I = {info_positions}")
							# print sorted info_positions
							print(f"Sorted info positions I : {np.sort(info_positions)}")
							logging.info(f"Sorted info positions I : {np.sort(info_positions)}")
			
							logging.info(f"Final sequence: {final_seq}")
	   
					if epoch % 1000 == 0:
						# save the model
						torch.save(setnet.state_dict(), f"{model_dir}/setnet_{sim_str}_{epoch}.pt")
						
						# save the model if the loss is the lowest so far
						if loss.item() < min_loss:
							min_loss = loss.item()
							min_seq = final_seq
							torch.save(setnet.state_dict(), f"{model_dir}/setnet_{sim_str}_best.pt")
							print(f"Best model saved at epoch {epoch}")
							logging.info(f"Best model saved at epoch {epoch}")

						# save the bler_lut periodically; saving after every epoch is too slow
						with open(f"{bler_lut_dir}/bler_{sim_str}.pkl", "wb") as f:
							pickle.dump(bler_lut, f)
						with open(f"{bler_lut_dir}/ber_{sim_str}.pkl", "wb") as f:
							pickle.dump(ber_lut, f)
			else:
				# load the model from the best epoch
				print(f"Running inference: Loading the best model from {model_dir}/setnet_{sim_str}_best.pt")
				logging.info(f"Running inference: Loading the best model from {model_dir}/setnet_{sim_str}_best.pt")
				setnet.load_state_dict(torch.load(f"{model_dir}/setnet_{sim_str}_best.pt"))
	
				# set the model to eval mode
				setnet.eval()

				with torch.no_grad():
					# test the model
					in_data = torch.tensor(rs_5g[:crc_len].copy()).to(device)
					# sample a sequence of indices
					probs_list, seq_list = setnet(in_data, 1, strategy='greedy')
			
					final_seq = seq_list[0]

					K_ref = Nt//2
					# compute BLER using py_aff3ct
					info_positions = final_seq[:K_ref].detach().cpu().numpy()
     
					# get the frozen_pos vector
					frozen_bits = np.ones(Nt, dtype=bool)
					frozen_bits[info_positions] = 0
	
					SNRs = np.array([ SNR_ref, ])
					bers, blers = polar_ber(Nt, K_ref, frozen_bits, SNRs, decoder_type, list_size, crc_len)

					# print final BLER
					print(f"Final rate 1/2 BLER: {blers[0]} at SNR = {SNRs[0]} with I = {info_positions}")
					logging.info(f"Final rate 1/2 BLER: {blers[0]} at SNR = {SNRs[0]} with I = {info_positions}")
					# print sorted info_positions
					print(f"Sorted info positions I : {np.sort(info_positions)}")
					logging.info(f"Sorted info positions I : {np.sort(info_positions)}")
	
	
					logging.info(f"Final sequence: {final_seq}")
	
	except KeyboardInterrupt:
	
		print("Saving the latest bler_lut")
		if 'bler_lut' in globals():
			# save the bler_lut after the full simulation
			with open(f"{bler_lut_dir}/bler_{sim_str}.pkl", "wb") as f:
				pickle.dump(bler_lut, f)
		if 'ber_lut' in globals():
			with open(f"{bler_lut_dir}/ber_{sim_str}.pkl", "wb") as f:
				pickle.dump(ber_lut, f)
	
		# save the model
		torch.save(setnet.state_dict(), f"{model_dir}/setnet_{sim_str}_{epoch}.pt")
		
		# save the model if the loss is the lowest so far
		if loss.item() < min_loss:
			min_loss = loss.item()
			min_seq = final_seq
			torch.save(setnet.state_dict(), f"{model_dir}/setnet_{sim_str}_best.pt")
			print(f"Best model saved at epoch {epoch}")
			logging.info(f"Best model saved at epoch {epoch}")
