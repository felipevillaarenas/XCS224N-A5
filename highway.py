#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1d


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils


class Highway(nn.Module):
    """ Highway Networks implementation (Srivastava et al., 2015):
        - have askip-connection controlled by a dynamic gate
    """

    def __init__(self, embed_size):
        """ Init Highway Model.

        @param embed_size (int): Word embedding size (dimensionality)
        """
        super(Highway, self).__init__()
        self.embed_size = embed_size
        self.projection = nn.Linear(self.embed_size, self.embed_size)
        self.gate       = nn.Linear(self.embed_size, self.embed_size)
    
    def forward(self, x_conv_out: torch.Tensor) -> torch.Tensor:
        """
        Map from x_conv_out to x_highway
        :param x_conv_out: Tensor output from cnn layer. Input size (batch_size, embedding_size)
        :return: x_highway: Tensor output from Highway network. Output size (batch_size, embedding_size)
        """
        # In the comments we’ll describe the dimensions for a single example (not a batch).
        # Then, sent_len and batch_size should be taking into account.
        # Highway layer.
            # x_proj = ReLU(W_proj x_conv_out + b_proj); ∈ R e_{word}
            # x_gate = σ(W_gate x_conv_out + b_gate); ∈ R e_{word}
            # x_highway = x_gate ⊙ x_proj + (1 − x_gate) ⊙ x_conv_out; ∈ R e_{word}
        x_projection = F.relu_(self.projection(x_conv_out))
        x_gate = torch.sigmoid(self.gate(x_conv_out))
        
        x_highway = x_gate * x_projection + (1 - x_gate) * x_conv_out
        return x_highway

### END YOUR CODE 

