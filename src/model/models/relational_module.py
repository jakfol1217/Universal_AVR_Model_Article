import pytorch_lightning as pl
import torch
import torch.nn as nn
from .vit import ViT
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


class RelationalModule(pl.LightningModule):
    def __init__(self,
                 cfg,
                 object_size: int,
                 asymetrical: bool,
                 rel_activation_func: str = "softmax",
                 context_norm: bool = True,
                 hierarchical: bool = False,
                 **kwargs
                 ):
        super(RelationalModule, self).__init__()

        self.object_size = object_size
        
        if asymetrical:
            self.k_trans = nn.Linear(object_size, object_size)
            self.q_trans = nn.Linear(object_size, object_size)
        else:
            self.k_trans = nn.Identity()
            self.q_trans = nn.Identity()


        if rel_activation_func == "softmax":
            self.rel_activation_func = nn.Softmax(dim=2)
        elif rel_activation_func == "tanh":
            self.rel_activation_func = nn.Tanh()
        else:
            self.rel_activation_func = nn.Identity() # todo: more activation functions?

        self.context_norm = context_norm # todo where to put this? probably at the start also what if multiple problems

        if hierarchical:
            self.create_relational = self.relational_bottleneck_hierarchical
            self.hierarchical_aggregation = nn.Parameter(data=torch.rand(2, requires_grad=True))  # creates weighted sum with learnable parameters
        else:
            self.create_relational = self.relational_bottleneck
            self.hierarchical_aggregation = nn.Parameter(data=torch.tensor([1,0], dtype=torch.float32, requires_grad=False))


        self.gamma = nn.Parameter(torch.ones(object_size))
        self.beta = nn.Parameter(torch.zeros(object_size))

    def apply_context_norm(self, z_seq):
        eps = 1e-6
        z_mu = z_seq.mean(1)
        z_sigma = (z_seq.var(1) + eps).sqrt()
        z_seq = (z_seq - z_mu.unsqueeze(1)) / z_sigma.unsqueeze(1)
        z_seq = (z_seq * self.gamma.unsqueeze(0).unsqueeze(0)) + self.beta.unsqueeze(
            0
        ).unsqueeze(0)
        return z_seq


    def forward(self, context: torch.Tensor, answers: torch.Tensor, *args) -> torch.Tensor:

        relational_matrices = []
        for ans_i in range(answers.shape[1]):
            context_choice = torch.cat([context, answers[:, ans_i, :].unsqueeze(1)], dim=1) # appending answer to context
            if self.context_norm:
                context_choice = self.apply_context_norm(context_choice)
            
#            if torch.any(context_choice.isinf()) or torch.any(context_choice.isnan()):
#                print("after normalization")
#                print(context_choice)

            keys = self.k_trans(context_choice) # creating keys
            queries = self.q_trans(context_choice) # creating queries
#            if torch.any(keys.isinf()) or torch.any(keys.isnan()):
#                print("keys")
#                print(keys)
#            if torch.any(queries.isinf()) or torch.any(queries.isnan()):
#                print("queries")
#                print(queries)
            rel_matrix_1, rel_matrix_2 = self.create_relational(keys, queries)

            rel_matrix = torch.cat([rel_matrix_1.unsqueeze(1), rel_matrix_2.unsqueeze(1)], dim=1)
            rel_matrix = torch.einsum('btch,m->bch', rel_matrix, self.hierarchical_aggregation) # aggregating 1st and 2nd degree realtional matrices
            relational_matrices.append(rel_matrix.unsqueeze(1))

        return torch.cat(relational_matrices, dim=1)




# todo: potentially other module for abstract shapes? utilizing slots etc

    def relational_bottleneck(self, keys, queries): # creating relational matrices of 1st degree only

        rel_matrix = torch.matmul(keys, queries.transpose(1,2))
#        rel_matrix = rel_matrix/torch.amax(rel_matrix, dim=(1,2)).unsqueeze(1).unsqueeze(1)
        return self.rel_activation_func(rel_matrix), torch.zeros(*rel_matrix.shape, device=rel_matrix.device)

    def relational_bottleneck_hierarchical(self, keys, queries): # creating relational matrices of 1st and 2nd degrees

        rel_matrix_1 = torch.matmul(keys, queries.transpose(1,2))

        rel_matrix_1 = self.rel_activation_func(rel_matrix_1)  # use activation on previous if hierarchical? probably not
        rel_matrix_2 = torch.matmul(rel_matrix_1, rel_matrix_1.transpose(1,2))
        return rel_matrix_1, self.rel_activation_func(rel_matrix_2)





class RelationalModuleSymAsym(pl.LightningModule):
    def __init__(self,
                 cfg,
                 object_size: int,
                 rel_activation_func: str = "none",
                 aggregate: bool = True,
                 context_norm: bool = False,
                 **kwargs
                 ):
        super(RelationalModuleSymAsym, self).__init__()

        self.aggregator = None
        self.object_size = object_size
        if aggregate:
            self.aggregator = nn.Parameter(data=torch.rand(2, requires_grad=True))
        self.rel_sym = RelationalModule(cfg=cfg,
                                        object_size=object_size,
                                        asymetrical=False,
                                        rel_activation_func=rel_activation_func,
                                        context_norm=context_norm,
                                        hierarchical=False)
        self.rel_asym = RelationalModule(cfg=cfg,
                                        object_size=object_size,
                                        asymetrical=True,
                                        rel_activation_func=rel_activation_func,
                                        context_norm=context_norm,
                                        hierarchical=False)
        

    def forward(self, context: torch.Tensor, answers: torch.Tensor, agg=None) -> torch.Tensor:

        rel_matrix = self.rel_sym(context, answers)

        rel_matrix_asym = self.rel_asym(context, answers)

    
        if self.aggregator is not None:
            aggregator = self.aggregator.softmax(-1)
            if agg is not None:
                aggregator = agg.softmax(-1)
            rel_mat_comb = torch.cat([rel_matrix.unsqueeze(2), rel_matrix_asym.unsqueeze(2)], dim=2)
            rel_mat_comb = torch.einsum('btdch,m->btch', rel_mat_comb, aggregator)

        else:
            rel_mat_comb = torch.cat([rel_matrix, rel_matrix_asym], dim=2)

        return rel_mat_comb





class RelationalScoringModule(pl.LightningModule):
    def __init__(self,
                 cfg,
                 in_dim:int,
                 hidden_dim: int = [256],
                 pooling: str = "max",
                 transformer: pl.LightningModule = None,
                 *args,
                 **kwargs
                 ):
        super(RelationalScoringModule, self).__init__()

#        self.scoring_mlp = nn.Sequential(
#                nn.Linear(in_dim, hidden_dim),
#                nn.ReLU(),
#                nn.Linear(hidden_dim, 1)
#        )
        try:
            len(hidden_dim)
        except:
            hidden_dim = [hidden_dim]

        self.scoring_mlp = LinearModule(
                    cfg=None,
                    in_dim=in_dim,
                    hidden_dims  = hidden_dim,
                    out_dim = 1,
        )


        if pooling == "max":
            self.pooling = nn.AdaptiveMaxPool2d((1, in_dim))
        elif pooling == "avg":
            self.pooling = nn.AdaptiveAvgPool2d((1, in_dim))
        else:
            raise ValueError(f"Pooling type {pooling} not supported")

        self.softmax = nn.Softmax(dim=1)


        if isinstance(transformer, (dict, DictConfig)):
            self.transformer=instantiate(transformer, cfg=None)
        else:
            self.transformer=transformer


    def forward(self, rel_matrix: torch.Tensor) -> torch.Tensor:
        answer_scores = []
        if self.transformer is None: # using MLP for scoring
            rel_matrix = rel_matrix.flatten(-2).unsqueeze(-2)
            rel_matrix = self.pooling(rel_matrix).squeeze(-2) # pooling -- to make relational matrices from different task types the same size (e.g. from bongard and analogy making)
            
            for ans_i in range(rel_matrix.shape[1]):
                answer_scores.append(self.scoring_mlp(rel_matrix[:, ans_i]))
            
        else: # using ViT transformer model for scoring
            for ans_i in range(rel_matrix.shape[1]):
                answer_scores.append(self.transformer(rel_matrix[:, ans_i]))
        answer_scores = torch.cat(answer_scores, dim=1)
        return answer_scores




class LinearModule(pl.LightningModule):
    def __init__(
        self,
        cfg,
        in_dim:int,
        hidden_dims  = [256],
        out_dim: int = 32,
        *args,
        **kwargs
    ):
        super().__init__()
        dims = [in_dim, *hidden_dims, out_dim]
        self.mlp = nn.Sequential(nn.Linear(dims[0], dims[1]))
        for i in range(1, len(dims[:-1])):
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Linear(dims[i], dims[i+1]))


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.mlp(input)


