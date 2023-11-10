import copy
import os, gc, torch
import time
import logging
#from huggingface_hub import hf_hub_download
from pynvml import *
from torch.nn import functional as F
import numpy as np
import json
from rwkv_tokenizer import TRIE_TOKENIZER
from sample_logits import sample_logits

from urllib.parse import urlsplit

from proxy_handler import proxy_handler
from rwkv_inference import rwkv_inference

# nvmlInit()
# gpu_h = nvmlDeviceGetHandleByIndex(0)
CONCURRENT_REQ_LIMIT = 50
CONCURRENT_REQ_COUNT = 0
ctx_limit = 8192
ctx_gpt_mode_chunks = 1024

#os.environ["CUDA_VISIBLE_DEVICES"] = ''
os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '0' # if '1' then use CUDA kernel for seq mode (much faster)

torch.set_num_threads(14)

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

current_dir = os.path.dirname( os.path.dirname(os.path.realpath(__file__)) )
model_path = current_dir + '/rwkv-3b-ai-town-v1.pth'

models = [
    RWKV(model=model_path, strategy='cpu fp32'),
]

pipelines = []


lockedModels = []

for model in models:
    pipelines.append(TRIE_TOKENIZER(current_dir + "/rwkv_vocab_v20230922_chatml.txt"))
    lockedModels.append(False)

# set thread count with pytorch

import asyncio

async def getModel():
    # while True:
    #     for i in range(len(models)):
    #         if not lockedModels[i]:
    #             #lockedModels[i] = True
    #             return models[i], pipelines[i], i
    #     await asyncio.sleep(0.1)
    return models[0], pipelines[0], 0

# def lockModel(i):
#     lockedModels[i] = True

# def unlockModel(i):
#     lockedModels[i] = False

def removeTokens(text):
    return text.replace("<|im_start|>", "").replace("<|im_end|>", "")

# dictionary of tuples of key => (state, expiration)
cachedStates = {}

async def buildPrompt(conversation, model, pipeline):
    first_message = conversation[0]
    fullprompt = f"<|im_start|>{first_message['role']}\n{removeTokens(first_message['content']).strip()}<|im_end|>\n"

    # # add system prompt to cache
    # cacheKey = hash(fullprompt)
    # if cacheKey not in cachedStates.keys():
    #     lockModel(0)
    #     input_tokens = pipeline.encode(fullprompt)[-ctx_limit:]
    #     statea = None
    #     for i in range(0, len(input_tokens), ctx_gpt_mode_chunks):
    #         out, statea = model.forward(input_tokens[i:min(ctx_gpt_mode_chunks, len(input_tokens)-i) + i], statea)
    #         gc.collect()
    #         del out

    #     statea = [state.cpu() for state in statea]
    #     unlockModel(0)
    #     cachedStates[hash(fullprompt)] = (statea, time.time() + 30) # cache 30 secs
            
    # hash current prompt to check for cached state
    state = None
    cacheKey = hash(fullprompt)

    prompt = fullprompt
    # if cacheKey in cachedStates.keys():
    #     state, expiration = cachedStates[cacheKey]
    #     prompt = ""
    #     # reset expiration
    #     cachedStates[cacheKey] = (state, time.time() + 60) # 1 minute
    #     state = copy.copy(state)
    #     state = [s.cpu() for s in state]
    #     print("## Using Cached State ##")
    
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
    typicalSampling = True
    
    prompt, statee, fullprompt = await buildPrompt(conversation, model, pipeline)
    
    full_response = fullprompt
    response = ""

    async for token, statee in rwkv_inference(prompt, model, pipeline, typicalSampling=typicalSampling, state=statee):
        full_response += token
        response += token
        yield token
        await asyncio.sleep(0.000001)

    CONCURRENT_REQ_COUNT -= 1

    
    print ("## Prompt ##")
    print (prompt)
    print ("## Response ##")
    print (response)
    
    print ("##################")
        
    cacheKey = full_response.strip() + "<|im_end|>\n"
    statee = [state.cpu() for state in statee]

    # cachedStates[hash(cacheKey)] = (statee, time.time() + 60 * 60) # cache state for 1 hour
    # gc.collect()
        

from aiohttp import web
import aiohttp
base_url = "https://api.openai.com/"

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

async def chat_handle(request):
    print("[CHAT] request started", request.path)

    # Concurrency locking
    global CONCURRENT_REQ_COUNT, CONCURRENT_REQ_LIMIT
    while CONCURRENT_REQ_COUNT >= CONCURRENT_REQ_LIMIT:
        await asyncio.sleep(0.01)
    CONCURRENT_REQ_COUNT += 1

    try:
        model, pipeline, index = await getModel()
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
            
        # unlockModel(index)

        await response.write("data: [DONE]\n\n".encode())
            
        print(f"## Time taken: {time.time() - startTime} ##")
        print(f"## Tokens generated: {totalTokens} ##")
        print(f"## Tokens per second: {totalTokens / (time.time() - startTime)} ##")
        
        await response.write_eof()
        return response
    except OSError:
        print("## Client disconnected ##")
        # unlockModel(index)
    finally:
        CONCURRENT_REQ_COUNT -= 1

app = web.Application()
logging.basicConfig(level=logging.DEBUG)

app.add_routes([
    web.post('/v1/chat/completions', chat_handle),
    web.post('/v1/embeddings', proxy_handler)
])

# ---
# async based background process
# ---
async def background_process():
    global CONCURRENT_REQ_COUNT, CONCURRENT_REQ_LIMIT
    while True:
        print(
            f"\n~~ Concurrent req: {CONCURRENT_REQ_COUNT}"
        )
        await asyncio.sleep(5)
    
def background_starter():
    asyncio.run(background_process())

threading.Thread(target=background_starter).start()

# THIS IS BLOCKING
web.run_app(app, port=9997)