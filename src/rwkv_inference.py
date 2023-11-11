
from rwkv_tokenizer import TRIE_TOKENIZER
from sample_logits import sample_logits
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
import os, gc, torch
import time, random, copy
ctx_gpt_mode_chunks = 1024

# Cache logic ?
global CACHE_ARR, CACHE_SIZE
CACHE_SIZE = 100
CACHE_ARR = [None]*CACHE_SIZE

# Perform prefix matching
def prefixMatching(long, short):
    short_len = len(short)
    # Find if the match failed
    for i in range(short_len):
        if long[i] != short[i]:
            return False
    # Finds a match!
    return True
    
# Get from cache, a given prompt_tokens
def getFromCache(in_prompt_tokens):
    global CACHE_ARR, CACHE_SIZE

    in_prompt_len = len(in_prompt_tokens)

    longest_match_len = 0
    longest_match_obj = None

    # Iterate all the cache items
    for i in range(CACHE_SIZE):
        sample_obj = CACHE_ARR[i]
        # Skip if empty
        if sample_obj == None:
            continue
        
        # Skip if cached tokens are "Larger" then the input
        sample_len = len(sample_obj["prompt_tokens"])
        if sample_len > in_prompt_len:
            continue
        if sample_len < longest_match_len:
            continue
        # Check if prefix match
        if prefixMatching(in_prompt_tokens, sample_obj["prompt_tokens"]) == False:
            continue
        
        # PREFIX matches
        longest_match_obj = sample_obj
        longest_match_len = sample_len
    
    # If there is no match, return the failure
    if longest_match_obj == None:
        return [], None, None, in_prompt_tokens

    # Return the matched token
    longest_match_obj["last_use"] = time.time()
    return (
        longest_match_obj["prompt_tokens"], 
        copy.deepcopy(longest_match_obj["logits"]), 
        copy.deepcopy(longest_match_obj["state"]), 
        in_prompt_tokens[len(longest_match_obj["prompt_tokens"]):] 
    )

# Insert into the cache
def setIntoCache(in_prompt_tokens, logits, state):
    global CACHE_ARR, CACHE_SIZE

    now = time.time()
    cache_obj = {
        "prompt_tokens" : in_prompt_tokens,
        "logits" : copy.deepcopy(logits),
        "state" : copy.deepcopy(state),
        "last_use" : now
    }
    
    oldest_entry_index = 0
    oldest_entry_time = now
    for i in range(random.randint(0,CACHE_SIZE/2) , CACHE_SIZE):
        sample_obj = CACHE_ARR[i]
        # Save immediately if empty
        if sample_obj == None:
            CACHE_ARR[i] = cache_obj
            return

        # find the oldest entry
        if sample_obj["last_use"] < now:
            oldest_entry_index = i
            oldest_entry_time = sample_obj["last_use"]
    
    # Finished loop, save it accordinlgy
    CACHE_ARR[oldest_entry_index] = cache_obj

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
    contextLengthWindow=8192 * 4,
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
    
    # Default out and state
    out = None
    state = None

    # Get from cache if possible
    cache_tokens, out, state, input_tokens = getFromCache( prompt_tokens[-contextLengthWindow:] )

    # Initial token gen (i==0)
    for i in range(0, len(input_tokens), ctx_gpt_mode_chunks):
        last_idx = min(ctx_gpt_mode_chunks, len(input_tokens)-i) + i
        out, state = model.forward(input_tokens[i:last_idx], state)

        # Store into cache
        if i == 0:
            joint_input = cache_tokens + input_tokens[:last_idx]
            setIntoCache(joint_input, out, state)

    # Additional token gen
    for i in range(int(token_count)):
        if i!=0:
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

