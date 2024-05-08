class Configs:

    def __init__(self,
                N_HEADS: int = 8,
                D_MODEL: int = 512,
                BATCH_SIZE: int = 64,
                SEQ_LEN: int = 100,
                D_HIDDEN: int = 2048,
                DROPOUT: int = 0.1,
                N_LAYERS: int = 3):
        
        self.n_heads = N_HEADS
        self.d_model = D_MODEL
        self.d_k = int(D_MODEL/N_HEADS)
        self.batch   = BATCH_SIZE
        self.seq_len =  SEQ_LEN
        self.d_hid   = D_HIDDEN
        self.dropout = DROPOUT
        self.n_layers = N_LAYERS