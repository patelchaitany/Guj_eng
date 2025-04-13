from dataclasses import dataclass
@dataclass
class Config:
    vocab_size:int  #vocabulary size
    d_model:int = 768 # Model dimension
    nhead:int = 8 # Number of heads in multihead attention
    num_encoder_layers:int = 6 #number of encoder layers
    num_decoder_layers:int = 6 #number of decoder layers
    dim_feedforward:int = 768 #dimension of feedforward network
    dropout:float = 0 #dropout probability
    max_len:int = 1024 #maximum length of input sequence
    num_embeddings_decoder:int = 768 #number of embeddings
    num_embeddings_encoder:int = 768 #number of embeddings
    comman_embedding_dim:int = 512 #common embedding dimension
    control_points:int = 20
