import torch
import torch.nn.functional as F

from torch.autograd import Variable
from torch import nn

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out))
    return out

class NatureCNN(nn.Module):
	def __init__(self, config):
		super(NatureCNN, self).__init__()
		self.stem = nn.Sequential(
			nn.Conv2d(4, 32, 8, stride=4),
			nn.ReLU(),
			nn.Conv2d(32, 64, 4, stride=2),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, stride=1),
			nn.ReLU()
		)
		self.hidden = nn.Linear(2304, 512)
		self.hidden.weight.data = normalized_columns_initializer(self.hidden.weight.data, 0.01)

	def forward(self, x):
		feats = self.stem(x)
		size = feats.size()
		feats = feats.view(-1, size[1] * size[2] * size[3])
		feats = F.relu(self.hidden(feats))
		return feats

class CNNPolicy(nn.Module):
	def __init__(self, config):
		super(CNNPolicy, self).__init__()
		self.feat_extractor = NatureCNN(config)
		self.output_dims = config.output_dims

		self.linear = nn.Linear(512, 256)
		self.pi = nn.Linear(256, self.output_dims)
		self.vf = nn.Linear(256, 1)

		self.pi.weight.data = normalized_columns_initializer(self.pi.weight.data, 0.01)
		self.vf.weight.data = normalized_columns_initializer(self.vf.weight.data, 1.0)

	def forward(self, x):
		feats = self.feat_extractor(x)
		feats = F.relu(self.linear(feats))
		logits = self.pi(feats)
		vf = self.vf(feats)
		return logits, vf
