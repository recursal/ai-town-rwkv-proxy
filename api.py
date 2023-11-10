import copy
import os, gc, torch
import time
#from huggingface_hub import hf_hub_download
from pynvml import *
from torch.nn import functional as F
import numpy as np
from urllib.parse import urlsplit
import json
from rwkv_tokenizer import TRIE_TOKENIZER
import aiohttp
# nvmlInit()
# gpu_h = nvmlDeviceGetHandleByIndex(0)
ctx_limit = 8192
ctx_gpt_mode_chunks = 64
concurrent_req_limit = 5
current_concurrent_req = 0
base_url = "https://api.openai.com/"
#os.environ["CUDA_VISIBLE_DEVICES"] = ''
os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1' # if '1' then use CUDA kernel for seq mode (much faster)

torch.set_num_threads(4)

from rwkv.model import RWKV
current_dir = os.path.dirname(os.path.realpath(__file__))
model_path = '/home/darok/Documents/GitHub/ai-town/rwkv-3b-ai-town-v1.pth'#current_dir + '/rwkv-ai-town-3b.pth'rwkv-3b-ai-town-v1.pth

models = [
    RWKV(model=model_path, strategy='cuda bf16'),
]

pipelines = []

from rwkv.utils import PIPELINE, PIPELINE_ARGS

lockedModels = []

for model in models:
    pipelines.append(TRIE_TOKENIZER(current_dir + "/rwkv_vocab_v20230922_chatml.txt"))
    lockedModels.append(False)

# set thread count with pytorch

import asyncio

async def getModel():
    while True:
        for i in range(len(models)):
            if not lockedModels[i]:
                #lockedModels[i] = True
                return models[i], pipelines[i], i
        await asyncio.sleep(0.1)

def lockModel(i):
    lockedModels[i] = True

def unlockModel(i):
    lockedModels[i] = False

def sample_logits_typical(logits, temperature=1.0, top_p=0.95, **kwargs):
        probs = F.softmax(logits.float(), dim=-1)
        logits = -torch.log(probs)
        ent = torch.nansum(logits * probs, dim=-1, keepdim=True)
        shifted_logits = torch.abs(logits - ent)
        sorted_ids = torch.argsort(shifted_logits)
        sorted_logits = shifted_logits[sorted_ids]
        sorted_probs = probs[sorted_ids]
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        cutoff = np.sum(cumulative_probs < top_p)
        probs[shifted_logits > sorted_logits[cutoff]] = 0
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        out = torch.multinomial(probs, num_samples=1)[0]
        return int(out)

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
        
async def evaluate(
    prompt,
    model,
    pipeline,
    token_count=1024,
    temperature=1.7,
    top_p=0.6,
    presencePenalty = 0.5,
    countPenalty = 0.5,
    typicalSampling = True,
    state = None
):
    args = PIPELINE_ARGS(temperature = max(0.2, float(temperature)), top_p = float(top_p),
                     alpha_frequency = countPenalty,
                     alpha_presence = presencePenalty,
                     token_ban = [], # ban the generation of some tokens
                     token_stop = [0, 65532]) # stop generation whenever you see any token here

    ctx = prompt
    
    # gpu_info = nvmlDeviceGetMemoryInfo(gpu_h)
    # print(f'vram {gpu_info.total} used {gpu_info.used} free {gpu_info.free}')
    
    all_tokens = []
    out_last = 0
    out_str = ''
    occurrence = {}
    for i in range(int(token_count)):
        if i==0:
            input_tokens = pipeline.encode(prompt)[-ctx_limit:]
            for i in range(0, len(input_tokens), ctx_gpt_mode_chunks):
                out, state = model.forward(input_tokens[i:min(ctx_gpt_mode_chunks, len(input_tokens)-i) + i], state)
                gc.collect()
            torch.cuda.empty_cache()
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
    torch.cuda.empty_cache()
    gc.collect()
    


# time the evaluation
# starttime = time.time()

# tokens_generated = 0
# run generator and output the result
# print("## Prompt ##")
# print(prompt)

# print("## Normal Sampling ##")
# for token in evaluate(prompt, model_type = model, typicalSampling=False):
#     print(token, end='', flush=True)
#     tokens_generated += 1

# print('\n')


# print("## Typical Sampling ##")
# for token in evaluate(prompt, model_type = model, typicalSampling=True):
#     print(token, end='', flush=True)
#     tokens_generated += 1


# print('\n')

def removeTokens(text):
    return text.replace("<|im_start|>", "").replace("<|im_end|>", "")

# dictionary of tuples of key => (state, expiration)
cachedStates = {}

async def buildPrompt(conversation, model, pipeline):
    first_message = conversation[0]
    fullprompt = f"<|im_start|>{first_message['role']}\n{removeTokens(first_message['content']).strip()}<|im_end|>\n"
    # add system prompt to cache
    cacheKey = hash(fullprompt)
    if cacheKey not in cachedStates.keys():
        lockModel(0)
        input_tokens = pipeline.encode(fullprompt)[-ctx_limit:]
        statea = None
        for i in range(0, len(input_tokens), ctx_gpt_mode_chunks):
            out, statea = model.forward(input_tokens[i:min(ctx_gpt_mode_chunks, len(input_tokens)-i) + i], statea)
            gc.collect()
            del out

        statea = [state.cpu() for state in statea]
        unlockModel(0)
        cachedStates[hash(fullprompt)] = (statea, time.time() + 30) # cache 30 secs
            
    # hash current prompt to check for cached state
    state = None
    cacheKey = hash(fullprompt)
    if cacheKey in cachedStates.keys():
        state, expiration = cachedStates[cacheKey]
        prompt = ""
        # reset expiration
        cachedStates[cacheKey] = (state, time.time() + 60) # 1 minute
        state = copy.copy(state)
        state = [s.cpu() for s in state]
        print("## Using Cached State ##")
    else:
        prompt = fullprompt
    
    for m in conversation[1:-1]:
        if m['role'] == 'user':
            fullprompt += "<|im_start|>user\n" + removeTokens(m['content']).strip() + "<|im_end|>\n"
            prompt += "<|im_start|>system\n" + removeTokens(m['content']).strip() + "<|im_end|>\n"
        elif m['role'] == 'assistant':
            fullprompt += "<|im_start|>assistant\n" + removeTokens(m['content']).strip() + "<|im_end|>\n"
            prompt += "<|im_start|>system\n" + removeTokens(m['content']).strip() + "<|im_end|>\n"
        elif m['role'] == 'system':
            fullprompt += "<|im_start|>system\n" + removeTokens(m['content']).strip() + "<|im_end|>\n"
            prompt += "<|im_start|>system\n" + removeTokens(m['content']).strip() + "<|im_end|>\n"
    
    # trim message
    last_message = conversation[-1]
            
    prompt += f"<|im_start|>{last_message['role']}\n" + removeTokens(last_message['content']).strip() + "<|im_end|>\n<|im_start|>assistant\n"
    fullprompt += f"<|im_start|>{last_message['role']}\n" + removeTokens(last_message['content']).strip() + "<|im_end|>\n<|im_start|>assistant\n"
    
    return prompt, state, fullprompt
    

async def handleRWKV(conversation, model, pipeline):
    global current_concurrent_req, concurrent_req_limit
    typicalSampling = True
    
    prompt, statee, fullprompt = await buildPrompt(conversation, model, pipeline)
    
    full_response = fullprompt
    response = ""

    while current_concurrent_req >= concurrent_req_limit:
        await asyncio.sleep(0.000001)


    current_concurrent_req += 1

    async for token, statee in evaluate(prompt, model, pipeline, typicalSampling=typicalSampling, state=statee):
        full_response += token
        response += token
        yield token
        await asyncio.sleep(0.000001)

    current_concurrent_req -= 1

    
    print ("## Prompt ##")
    print (prompt)
    print ("## Response ##")
    print (response)
    
    print ("##################")
        
    cacheKey = full_response.strip() + "<|im_end|>\n"
    statee = [state.cpu() for state in statee]
    cachedStates[hash(cacheKey)] = (statee, time.time() + 60 * 60) # cache state for 1 hour
    gc.collect()
        

from aiohttp import web
import logging

async def buildOutputChunk(token):
    object = {
        'object': 'chat.completion.chunk',
        'choices': [
            {
              "delta": {
                "content": token
              },
              "finish_reason": None,
              "index": 0
            }
        ],
    }
    return "data: " + json.dumps(object) + "\n\n"

async def proxy_embedding(request):

    url = base_url + request.path
    
    method = request.method
    headers = dict(request.headers)

    headers['Host'] = "{0.netloc}".format(urlsplit(base_url))

    log_completion = False

    if "chat/completions" in url:
        log_completion = True
        print("Completion request detected")
    else:
        print("Non-completion request detected")
        print(url)

    req_text = await request.text()
    log_summary = False

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=req_text) as response:

            headers = {k: v for k, v in response.headers.items()}
            
            full_response = ""
            if isinstance(response, web.StreamResponse):
                # Stream response back
                response = web.StreamResponse(
                    status=response.status,
                    headers=headers,
                )
                response.content_length = response.content_length
                
                async for data in response.content.iter_any():
                    full_response += data
                    await response.write(data)
            else:
                full_response = await response.text()
                return web.Response(text=full_response)

async def handle(request):
    model, pipeline, index = await getModel()
    try:
        response = web.StreamResponse(
            status=200,
            reason='OK',
            headers={'Content-Type': 'text/plain'},
        )
        await response.prepare(request)
        # get the request data (json)
        data = await request.json()    
        
        startTime = time.time()
        totalTokens = 0
        
        # run handleRwkv generator and output the result
        async for token in handleRWKV(data['messages'], model, pipeline):
            await response.write((await buildOutputChunk(token)).encode())
            await asyncio.sleep(0.000001)
            totalTokens += 1
            
        unlockModel(index)

        await response.write("data: [DONE]\n\n".encode())
            
        print(f"## Time taken: {time.time() - startTime} ##")
        print(f"## Tokens generated: {totalTokens} ##")
        print(f"## Tokens per second: {totalTokens / (time.time() - startTime)} ##")
        
        await response.write_eof()
        return response
    except OSError:
        print("## Client disconnected ##")
        unlockModel(index)

app = web.Application()
logging.basicConfig(level=logging.DEBUG)
app.add_routes([
    web.post('/v1/chat/completions', handle),
    web.post('/v1/embeddings', proxy_embedding)
])

def cleanCachedStates():
    while True:
        time.sleep(15) # every minute
        for k in cachedStates.keys():
            if cachedStates[k][1] < time.time():
                del cachedStates[k]
                print("## Cleared A Cached State ##")
                break
            
threading.Thread(target=cleanCachedStates).start()

web.run_app(app, port=9997)
