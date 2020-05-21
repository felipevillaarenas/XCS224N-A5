#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn


# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

from cnn import CNN
from highway import Highway

CNN_KERNEL   = 5
CHAR_EMBED   = 50
MAX_WORD_LEN = 21
DROPOUT_RATE = 0.3
# End "do not change" 

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output. In the PDF e_{word} 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        self.max_word_length = MAX_WORD_LEN          # m_{word}
        self.embed_size      = embed_size   # e_{word}, Also e_{word} = f.
        self.vocab           = vocab
        self.char_embed_size = CHAR_EMBED           # e_{char}
        self.kernel_size     = CNN_KERNEL
        self.dropout_rate    = DROPOUT_RATE

        self.char_embedding = nn.Embedding(
            num_embeddings  = len(vocab.char2id), 
            embedding_dim   = self.char_embed_size, 
            padding_idx     = vocab.char2id['<pad>']
        )

        self.CNN = CNN(
            char_embed_size  = self.char_embed_size,
            filters = self.embed_size, 
            max_word_length = self.max_word_length,
            kernel_size  = self.kernel_size
        )

        self.Highway = Highway(embed_size=self.embed_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        ### END YOUR CODE

    def forward(self, x_padded):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param x_padded: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param x_word_emb: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        # In the comments we’ll describe the dimensions for a single example (not a batch).
        # Then, sent_len and batch_size should be taking into account.
        # I. Padding and embedding lookup. 
            # x_{emb} = CharEmbedding(x_{padded}) ; R m_{word} x e_{char} 
            # x_reshaped =Reshape(x_{emb}); R e_{char}xm_{word} 
        x_emb= self.char_embedding(x_padded)
        sent_len, batch_size, m_word, e_char = x_emb.shape
        x_reshaped = x_emb.view((sent_len * batch_size, m_word, self.char_embed_size)).permute(0,2,1)                                     
        
        # II. Convolutional network. 
            # x_conv = Conv1D(x_reshaped); ∈ R e_{word}x(m_{word}−k+1)
            # x_conv_out = MaxPool(ReLU(xconv)); ∈ R e_{word}
            # in our implementation e_{word} is equal to the number of filters f.
        x_conv_out = self.CNN(x_reshaped)
        

        # III. Highway layer.
            # x_proj = ReLU(W_proj x_conv_out + b_proj); ∈ R e_{word}
            # x_gate = σ(W_gate x_conv_out + b_gate); ∈ R e_{word}
            # x_highway = x_gate ⊙ x_proj + (1 − x_gate) ⊙ x_conv_out; ∈ R e_{word}
        x_highway = self.Highway(x_conv_out)

        # IV. Dropout.
            # x_word_emb = Dropout(x_highway); ∈ R e_{word}
        x_word_emb = self.dropout(x_highway)
        x_word_emb = x_word_emb.view(sent_len, batch_size, self.embed_size)
        
        return x_word_emb
        ### END YOUR CODE
