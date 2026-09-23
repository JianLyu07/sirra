import init_utils

import torch
import torch.nn.functional as F
from torch import Tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
def get_mean_pooling_vec(last_hidden_state,attention_mask):
    #由模型的last_hidden_state和attention_mask计算mean_pooling向量
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    sentence_embeddings = sum_embeddings / sum_mask
    normalized_sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return normalized_sentence_embeddings

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    #由模型的last_hidden_state和attention_mask计算last_token_pooling向量
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        sentence_embeddings = last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        sentence_embeddings = last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    normalized_sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return normalized_sentence_embeddings