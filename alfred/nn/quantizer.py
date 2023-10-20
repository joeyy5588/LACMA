import torch
import torch.nn as nn

class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta=0.25):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        weight_fn = 'ET/logs/codebook_{}.pt'.format(self.n_e)
        init_weight = torch.load(weight_fn)
        init_weight = nn.Parameter(init_weight, requires_grad=True)
        self.embedding.weight = init_weight
        self.register_buffer('count', torch.zeros(1), persistent=False)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        d = torch.sum(z ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z, self.embedding.weight.t())
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        if self.count % 100 == 0:
            print('code num', torch.unique(min_encoding_indices).shape[0])
        self.count += 1
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)
        
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        z_q = z + (z_q - z).detach()

        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        return z_q, perplexity


class VectorQuantizerRestart(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE with random restart.
    After every epoch, run:
    random_restart()
    reset_usage()
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    - usage_threshold : codes below threshold will be reset to a random code
    """

    def __init__(self, n_e=1024, e_dim=256, beta=0.25, usage_threshold=1.0e-9):
        super().__init__()

        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.usage_threshold = usage_threshold

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        weight_fn = 'ET/logs/codebook_{}.pt'.format(self.n_e)
        init_weight = torch.load(weight_fn)
        init_weight = nn.Parameter(init_weight, requires_grad=True)
        self.embedding.weight = init_weight
        
        # initialize usage buffer for each code as fully utilized
        self.register_buffer('usage', torch.ones(self.n_e), persistent=False)
        
        self.perplexity = None
        self.loss = None

    def dequantize(self, z):
        z_flattened = z.view(-1, self.e_dim)
        z_q = self.embedding(z_flattened).view(z.shape)
        return z_q

    def update_usage(self, min_enc):
        self.usage[min_enc] = self.usage[min_enc] + 1  # if code is used add 1 to usage
        self.usage /= 2 # decay all codes usage
        print('usage', torch.sum(self.usage > 0.1))

    def reset_usage(self):
        self.usage.zero_() #  reset usage between epochs

    def random_restart(self):
        #  randomly restart all dead codes below threshold with random code in codebook
        dead_codes = torch.nonzero(self.usage < self.usage_threshold).squeeze(1)
        rand_codes = torch.randperm(self.n_e)[0:len(dead_codes)]
        with torch.no_grad():
            self.embedding.weight[dead_codes] = self.embedding.weight[rand_codes]

    def forward(self, z, return_indices=False):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        print('code num', torch.unique(min_encoding_indices).shape[0])
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).type_as(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight)
        z_q = z_q.view(z.shape)

        self.update_usage(min_encoding_indices)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        self.perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        return z_q, self.perplexity


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        weight_fn = 'ET/logs/codebook_{}.pt'.format(num_embeddings)
        init_weight = torch.load(weight_fn)
        init_weight = nn.Parameter(init_weight, requires_grad=True)
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight = init_weight
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('usage', torch.ones(self._num_embeddings), persistent=False)
        self.register_buffer('count', torch.zeros(1), persistent=False)
        self._ema_w = init_weight
        # self._ema_w.data.normal_()

              
        self._decay = decay
        self._epsilon = epsilon

    def update_usage(self, min_enc):
        self.usage[min_enc] = self.usage[min_enc] + 1

    def print_usage(self):
        # sort usage
        usage, indices = torch.sort(self.usage, descending=True)
        print("top indices", indices[:30])


    def forward(self, z):
        # Calculate distances
        distances = (torch.sum(z**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(z, self._embedding.weight.t()))
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        if self.count % 100 == 0:
            print('code num', torch.unique(encoding_indices).shape[0])
        self.count += 1
        self.update_usage(encoding_indices)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(z.shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), z)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
                
        # Straight Through Estimator
        quantized = z + (quantized - z).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return quantized, perplexity