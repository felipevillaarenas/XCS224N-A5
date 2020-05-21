#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1e

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils

class CNN(nn.Module):
    """1-dimensional convolutions used to combine the character embeddings. 
    The convolutional layer has two hyperparameters: the kernel size k (also 
    called window size), which dictates the size of the window used to compute features, 
    and the number of filters f.
    """
    def __init__(self, char_embed_size, filters, max_word_length, kernel_size):
        """Init CNN Module.

        @param char_embed_size (int): embedding size (dimensionality) of each character in a word
        @param filters (int): number of output channels or embed_size
        @param max_word_length (int): maximum length of a word
        @param kernel_size (int): window size used to compute features
        """
        super(CNN,self).__init__()
        self.char_embed_size    = char_embed_size
        self.filters            = filters
        self.max_word_length    = max_word_length
        self.kernel_size        = kernel_size

        self.conv1d = nn.Conv1d(
            in_channels=char_embed_size,
            out_channels=filters,
            kernel_size=kernel_size,
            bias=True
        )
        self.max_pool_1d = nn.MaxPool1d(max_word_length - kernel_size + 1)

    def forward(self, x_reshaped):
        """
        Compute word embedding
        @param input (Tensor): shape (batch_size, char_embed_size, max_word_length)
        @return (Tensor): shape (batch_size, embed_size), word embedding of each word in batch
        """
        # In the comments we’ll describe the dimensions for a single example (not a batch).
        # Then, sent_len and batch_size should be taking into account to reshape the tensor before the 
        # convolutional stage and after the dropout layer.

        # Convolutional network. 
            # x_conv = Conv1D(x_reshaped); ∈ R e_{word}x(m_{word}−k+1)
            # x_conv_out = MaxPool(ReLU(xconv)); ∈ R e_{word}
            # in our implementation e_{word} is equal to the number of filters f.
        x_conv = self.conv1d(x_reshaped)
        x_conv_out = self.max_pool_1d(F.relu_(x_conv)).squeeze()
        return x_conv_out
### END YOUR CODE

