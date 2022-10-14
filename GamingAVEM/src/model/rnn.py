import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class SelfAttention(nn.Module):
	"""
		Self-attention layer

	Args:
		attention_size: int, attention vector length
		batch_first: bool, affects tensor ordering of dims
		layers: int, num of attention layers
		dropout: float, dropout rate
		non_linearity: str, activation function
		
	"""
	def __init__(self, attention_size, batch_first=True, layers=1,
		dropout=0, non_linearity="tanh"):
		super(SelfAttention, self).__init__()
		self.batch_first = batch_first
		if non_linearity == "relu":
			activation = nn.ReLU()
		elif non_linearity == "tanh":
			activation = nn.Tanh()
		else:
			raise KeyError("Undefined activation function!")

		modules = []
		for i in range(layers - 1):
			modules.append(nn.Linear(attention_size, attention_size))
			modules.append(activation)
			modules.append(nn.Dropout(dropout))

		modules.append(nn.Linear(attention_size, 1))
		modules.append(activation)
		modules.append(nn.Dropout(dropout))

		self.attention = nn.Sequential(*modules)
		self.softmax = nn.Softmax(dim=-1)

	@staticmethod
	def get_mask(attentions, lengths):
		"""
			Construct mask for padded items from lengths

		Args:
			attentions: torch.Tensor
			lengths: torch.Tensor

		Return:

		"""
		max_len = max(lengths.data)
		mask = Variable(torch.ones(attentions.size())).detach()
		mask = mask.to(DEVICE)
		for i, l in enumerate(lengths.data):
			if l < max_len:
				mask[i, l:] = 0
		return mask

	def forward(self, x, lengths):
		"""
			Forward pass in self-attention.
			Steps:
				- dot product <attention, hidden state>
				- masking by length
				- Weighted sum of scores

		Args:
			x: (B, L, H) torch.Tensor of input sequence vectors
			lengths: 

		Return:
			a (B, H) torch.Tensor of weighted vector values
			a (B, H) torch.Tensor of attention values
		"""
		scores = self.attention(x).squeeze(-1)
		scores = self.softmax(scores)

		mask = self.get_mask(scores, lengths)
		masked_scores = scores * mask
		sums_ = masked_scores.sum(-1, keepdim=True)
		scores = masked_scores.div(sums_)
		
		weighted = torch.mul(x, scores.unsqueeze(-1).expand_as(x))
		representations = weighted.sum(1).squeeze(-1)

		return representations, scores

class SequenceSorter:
	"""
		Sort batch data and labels by sequence length
	
	Args:
		lengths: nn.Tensor, lengths of the data sequences

	Return:
		a nn.Tensor or sorted lengths
		a callable method that sorts iterable items
		a callable method that unsorts iterable items to original
			order
	"""
	@staticmethod
	def _sort_by(lengths):
		batch_size = lengths.size(0)
		sorted_lengths, sorted_idx = lengths.sort()
		_, original_idx = sorted_idx.sort(0, descending=True)
		reverse_idx = torch.linspace(batch_size - 1, 0, batch_size).long()
		reverse_idx = reverse_idx.to(DEVICE)
		sorted_lengths = sorted_lengths[reverse_idx]

		def sort(iterable):
			if len(iterable.shape) > 1:
				return iterable[sorted_idx.data][reverse_idx]
			return iterable

		def unsort(iterable):
			if len(iterable.shape) > 1:
				return iterable[reverse_idx][original_idx][reverse_idx]
			return iterable

		return sorted_lengths, sort, unsort

class LstmEncoder(nn.Module):
	"""
		A simple LSTM-cell-based encoder layer class

	Args:
		input_size: int, input dim
		rnn_size: int, LSTM cell h vector size
		num_layers: int, num of RNN layers
		dropout: dropout rate
		bidirectional: bool, whether layer is bidirectional

	"""
	def __init__(self, input_size, rnn_size, num_layers, dropout,
		bidirectional, batch_first=True):
		super(LstmEncoder, self).__init__()
		self.lstm = nn.LSTM(input_size=input_size,
			hidden_size=rnn_size,
			num_layers=num_layers,
			dropout=dropout,
			bidirectional=bidirectional,
			batch_first=True)
		self.dropout = nn.Dropout(dropout)
		self.feature_size = rnn_size
		if bidirectional:
			self.feature_size *= 2
		self.batch_first = batch_first
		self.num_layers = num_layers

	@staticmethod
	def last_by_index(outputs, lengths):
		"""
			Index of the last output for every sequence

		Args:
			outputs: 
			lengths:

		Return:

		"""
		idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
			outputs.size(2)).unsqueeze(1)
		return outputs.gather(1, idx).squeeze()

	@staticmethod
	def split_directions(outputs):
		"""
			Tell forward from backward outputs within a
			bidirectional layer

		Args:
			outputs:

		Return:
			a (B, L, H) torch.Tensor from forward LSTM pass
			a (B, L, H) torch.Tensor from backward LSTM pass
		"""
		direction_size = int(outputs.size(-1) / 2)
		forward = outputs[:, :, :direction_size]
		backward = outputs[:, :, direction_size:]
		return forward, backward

	def last_timestep(self, outputs, lengths, bi=False):
		"""
			Get exclusively the last output h_T

		Args:
			outputs:
			lengths:
			bi:

		Return:

		"""
		if bi:
			forward, backward = self.split_directions(outputs)
			last_forward = self.last_by_index(forward, lengths)
			if len(last_forward.size()) < 2:
				last_forward = last_forward.unsqueeze(0)
			last_backward = backward[:, 0, :]
			return torch.cat((last_forward, last_backward), dim=-1)

		return self.last_by_index(outputs, lengths)

	def forward(self, x, lengths):
		"""
			Forward module pass

		Args:
			x:
			lengths:

		Return:
			a (B, L, H) torch.Tensor of output features for every h_t
			a (B, H) torch.Tensor with the last output h_L
		"""
		packed = pack_padded_sequence(x, list(lengths.data),
			batch_first=self.batch_first)
		out_packed, _ = self.lstm(packed)
		outputs, _ = pad_packed_sequence(out_packed, 
			batch_first=self.batch_first)
		last_outputs = self.last_timestep(outputs, lengths, 
			self.lstm.bidirectional)
		# Force dropout if there s only 1 layer
		if self.num_layers < 2:
			last_outputs = self.dropout(last_outputs)
		return outputs, last_outputs

class LstmModal(nn.Module, SequenceSorter):
	"""
		Sequential model aimed at uni-modal problems,
		based on a encoder-decoder structure. 

		The encoder is conformed by a stack of (bi)LSTM cells.
		The decoder is made by a self-attention layer and a Linear
		classifier.

		Args:
		out_size: int, output layer's shape
		NOTE: all *_params required to follow above key ordering
	"""
	def __init__(self, out_size, input_size, **kwargs):
		super(LstmModal, self).__init__()

		self.encoder = LstmEncoder(input_size=input_size,
			rnn_size=50,
			num_layers=2,
			bidirectional=True,
			dropout=0.3)
		self.feature_size = self.encoder.feature_size # rnn_size

		self.attention = SelfAttention(
			attention_size=self.feature_size,
			layers=2,
			dropout=0.3,
			non_linearity="tanh")

		self.classifier = nn.Linear(in_features=self.feature_size, out_features=out_size)

	def forward(self, x, lengths):
		"""
			Forward pass through the network

		Args:
			x: nn.Tensor, sequences of data items
			lengths: nn.Tensor, sequences' lenghts

		Return:
			a (B, out_size) nn.Tensor, class logits
			a (B, encoder_dim) nn.Tensor, attention values
		"""
		lengths, sort, unsort = self._sort_by(lengths)
		x = sort(x)

		outputs, last_output = self.encoder(x.float(), lengths)
		attentions = None
		representations = last_output		
		representations, attentions = self.attention(outputs, lengths)

		representations = unsort(representations)
		if attentions is not None:
			attentions = unsort(attentions)
		logits = self.classifier(representations)
		return logits, attentions

class GruEncoder(nn.Module):
	"""
		A simple GRU-cell-based encoder layer class

	Args:
		input_size: int, input dim
		rnn_size: int, GRU cell h vector size
		num_layers: int, num of RNN layers
		dropout: dropout rate
		bidirectional: bool, whether layer is bidirectional

	"""
	def __init__(self, input_size, rnn_size, num_layers, dropout,
		bidirectional, batch_first=True):
		super(GruEncoder, self).__init__()
		self.gru = nn.GRU(input_size=input_size,
			hidden_size=rnn_size,
			num_layers=num_layers,
			dropout=dropout,
			bidirectional=bidirectional,
			batch_first=True)
		self.dropout = nn.Dropout(dropout)
		self.feature_size = rnn_size
		if bidirectional:
			self.feature_size *= 2
		self.batch_first = batch_first
		self.num_layers = num_layers

	@staticmethod
	def last_by_index(outputs, lengths):
		"""
			Index of the last output for every sequence

		Args:
			outputs: 
			lengths:

		Return:

		"""
		idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
			outputs.size(2)).unsqueeze(1)
		return outputs.gather(1, idx).squeeze()

	@staticmethod
	def split_directions(outputs):
		"""
			Tell forward from backward outputs within a
			bidirectional layer

		Args:
			outputs:

		Return:
			a (B, L, H) torch.Tensor from forward GRU pass
			a (B, L, H) torch.Tensor from backward GRU pass
		"""
		direction_size = int(outputs.size(-1) / 2)
		forward = outputs[:, :, :direction_size]
		backward = outputs[:, :, direction_size:]
		return forward, backward

	def last_timestep(self, outputs, lengths, bi=False):
		"""
			Get exclusively the last output h_T

		Args:
			outputs:
			lengths:
			bi:

		Return:

		"""
		if bi:
			forward, backward = self.split_directions(outputs)
			last_forward = self.last_by_index(forward, lengths)
			if len(last_forward.size()) < 2:
				last_forward = last_forward.unsqueeze(0)
			last_backward = backward[:, 0, :]
			return torch.cat((last_forward, last_backward), dim=-1)

		return self.last_by_index(outputs, lengths)

	def forward(self, x, lengths):
		"""
			Forward module pass

		Args:
			x:
			lengths:

		Return:
			a (B, L, H) torch.Tensor of output features for every h_t
			a (B, H) torch.Tensor with the last output h_L
		"""
		packed = pack_padded_sequence(x, list(lengths.data),
			batch_first=self.batch_first)
		out_packed, _ = self.gru(packed)
		outputs, _ = pad_packed_sequence(out_packed, 
			batch_first=self.batch_first)
		last_outputs = self.last_timestep(outputs, lengths, 
			self.gru.bidirectional)
		# Force dropout if there s only 1 layer
		if self.num_layers < 2:
			last_outputs = self.dropout(last_outputs)
		return outputs, last_outputs

class GruModal(nn.Module, SequenceSorter):
	"""
		Sequential model aimed at uni-modal problems,
		based on a encoder-decoder structure. 

		The encoder is conformed by a stack of (bi)LSTM cells.
		The decoder is made by a self-attention layer and a Linear
		classifier.

		Args:
		out_size: int, output layer's shape
		NOTE: all *_params required to follow above key ordering
	"""
	def __init__(self, out_size, input_size, **kwargs):
		super(GruModal, self).__init__()

		self.encoder = GruEncoder(input_size=input_size,
			rnn_size=50,
			num_layers=2,
			bidirectional=True,
			dropout=0.3)
		self.feature_size = self.encoder.feature_size # rnn_size

		self.attention = SelfAttention(
			attention_size=self.feature_size,
			layers=2,
			dropout=0.3,
			non_linearity="tanh")

		self.classifier = nn.Linear(in_features=self.feature_size, out_features=out_size)

	def forward(self, x, lengths):
		"""
			Forward pass through the network

		Args:
			x: nn.Tensor, sequences of data items
			lengths: nn.Tensor, sequences' lenghts

		Return:
			a (B, out_size) nn.Tensor, class logits
			a (B, encoder_dim) nn.Tensor, attention values
		"""
		lengths, sort, unsort = self._sort_by(lengths)
		x = sort(x)

		outputs, last_output = self.encoder(x.float(), lengths)
		attentions = None
		representations = last_output		
		representations, attentions = self.attention(outputs, lengths)

		representations = unsort(representations)
		if attentions is not None:
			attentions = unsort(attentions)
		logits = self.classifier(representations)
		return logits, attentions