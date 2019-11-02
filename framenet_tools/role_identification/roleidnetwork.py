import logging
import torch
import torch.nn as nn
import torchtext
import os

from torch.autograd import Variable
from tqdm import tqdm
from tensorboardX import SummaryWriter
from typing import List

from framenet_tools.config import ConfigManager


class Net(nn.Module):

    def __init__(self, embedding_layer: torch.nn.Embedding, device: torch.device):
        super(Net, self).__init__()


class RoleIdNetwork(object):
    def __init__(self, cM: ConfigManager, embedding_layer: torch.nn.Embedding):
        self.cM = cM
        self.embedding_layer = embedding_layer

        # Check for CUDA
        use_cuda = self.cM.use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        logging.debug(f"Device used: {self.device}")

    def train_network(self, train_set, dev_set):

        # Stub
        return
