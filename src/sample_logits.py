
import os, gc, torch
from torch.nn import functional as F
import numpy as np

# def sample_logits_typical(logits, temperature=1.0, top_p=0.95, **kwargs):
#         probs = F.softmax(logits.float(), dim=-1)
#         logits = -torch.log(probs)
#         ent = torch.nansum(logits * probs, dim=-1, keepdim=True)
#         shifted_logits = torch.abs(logits - ent)
#         sorted_ids = torch.argsort(shifted_logits)
#         sorted_logits = shifted_logits[sorted_ids]
#         sorted_probs = probs[sorted_ids]
#         cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
#         cutoff = np.sum(cumulative_probs < top_p)
#         probs[shifted_logits > sorted_logits[cutoff]] = 0
#         if temperature != 1.0:
#             probs = probs ** (1.0 / temperature)
#         out = torch.multinomial(probs, num_samples=1)[0]
#         return int(out)

def sample_logits(logits, temperature=1.0, top_p=0.85, top_k=0):
    probs = F.softmax(logits.float(), dim=-1)
    top_k = int(top_k)
    if probs.device == torch.device('cpu'):
        probs = probs.numpy()
        sorted_ids = np.argsort(probs)
        sorted_probs = probs[sorted_ids][::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
        probs[probs < cutoff] = 0
        if top_k < len(probs) and top_k > 0:
            probs[sorted_ids[:-top_k]] = 0
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        probs = probs / np.sum(probs)
        out = np.random.choice(a=len(probs), p=probs)
        return int(out)
    else:
        sorted_ids = torch.argsort(probs)
        sorted_probs = probs[sorted_ids]
        sorted_probs = torch.flip(sorted_probs, dims=(0,))
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
        probs[probs < cutoff] = 0
        if top_k < len(probs) and top_k > 0:
            probs[sorted_ids[:-top_k]] = 0
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        out = torch.multinomial(probs, num_samples=1)[0]
        return int(out)