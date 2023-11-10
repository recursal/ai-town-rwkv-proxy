
from rwkv_tokenizer import TRIE_TOKENIZER
from sample_logits import sample_logits
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
import os, gc, torch
ctx_gpt_mode_chunks = 1024

# Cache logic ?

# Perform inference or a given prompt / state
# including sampler selection, and presence penalty
@torch.no_grad()
async def rwkv_inference_tokens(
    prompt_tokens,
    model,
    pipeline,
    token_count=1024,
    temperature=1.7,
    top_p=0.6,
    presencePenalty = 0.5,
    countPenalty = 0.5,
    typicalSampling = True,
    contextLengthWindow=8192,
    state = None
):
    # Skip for empty request
    if token_count == 0:
        yield out_str, state
        return
    
    # Continue with everyhting else
    args = PIPELINE_ARGS(temperature = max(0.2, float(temperature)), top_p = float(top_p),
                     alpha_frequency = countPenalty,
                     alpha_presence = presencePenalty,
                     token_ban = [], # ban the generation of some tokens
                     token_stop = [0, 65532]) # stop generation whenever you see any token here

    # gpu_info = nvmlDeviceGetMemoryInfo(gpu_h)
    # print(f'vram {gpu_info.total} used {gpu_info.used} free {gpu_info.free}')
    
    all_tokens = []
    out_last = 0
    out_str = ''
    occurrence = {}

    for i in range(int(token_count)):
        if i==0:
            input_tokens = prompt_tokens[-contextLengthWindow:]
            for i in range(0, len(input_tokens), ctx_gpt_mode_chunks):
                out, state = model.forward(input_tokens[i:min(ctx_gpt_mode_chunks, len(input_tokens)-i) + i], state)
        else:
            out, state = model.forward([token], state)
            
        for n in occurrence:
            out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)
            
        # if typicalSampling:
        #     token = sample_logits_typical(out, temperature=args.temperature, top_p=args.top_p)
        # else:
        #     token = sample_logits(out, temperature=args.temperature, top_p=args.top_p)
        
        token = sample_logits(out, temperature=args.temperature, top_p=args.top_p)

        if token in args.token_stop:
            tmp = pipeline.decode(all_tokens[out_last:])
            yield tmp, state
            del out
            break
        all_tokens += [token]
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1
        
        tmp = pipeline.decode(all_tokens[out_last:])
        if '\ufffd' not in tmp:
            out_str += tmp
            yield tmp, state
            del out
            out_last = i + 1
        else:
            del out
    del occurrence

