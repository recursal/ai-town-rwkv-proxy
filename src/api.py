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
import random
from proxy_handler import proxy_handler
from rwkv_inference import rwkv_inference, rwkv_inference_tokens

global CONCURRENT_REQ_COUNT, CONCURRENT_REQ_LIMIT
CONCURRENT_REQ_COUNT = 0
CONCURRENT_REQ_LIMIT = 50

global INFERENCE_TIME_TAKEN_S, INFERENCE_PROMPT_TOKENS, INFERENCE_OUTPUT_TOKENS
INFERENCE_TIME_TAKEN_S = 0.001
INFERENCE_PROMPT_TOKENS = 0
INFERENCE_OUTPUT_TOKENS = 0

# nvmlInit()
# gpu_h = nvmlDeviceGetHandleByIndex(0)
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
    RWKV(model=model_path, strategy='cpu fp32')
]
pipelines = []
for model in models:
    pipelines.append(TRIE_TOKENIZER(current_dir + "/rwkv_vocab_v20230922_chatml.txt"))

# set thread count with pytorch

import asyncio

async def getModel():
    if len(models) > 0:
        i = int(random.random() * len(models))
    else:
        i = 0
    return models[i], pipelines[i], i

# dictionary of tuples of key => (state, expiration)
cachedStates = {}

def removeTokens(text):
    return text.replace("<|im_start|>", "").replace("<|im_end|>", "")

async def buildPrompt(conversation):
    # Build the prompt accordingly
    fullprompt = ""
    for m in conversation:
        fullprompt += "<|im_start|>"+ m["role"] +"\n" + removeTokens(m['content']).strip() + "<|im_end|>\n"
    # Start of assistant response
    fullprompt += "<|im_start|>assistant\n"
    return fullprompt

async def handleRWKV(conversation, model, pipeline):
    global INFERENCE_PROMPT_TOKENS, INFERENCE_OUTPUT_TOKENS
    typicalSampling = True
    
    fullprompt = await buildPrompt(conversation)
    fullprompt_tokens = pipeline.encode(fullprompt)

    statee = None

    full_response = fullprompt
    response = ""

    token_count = 0
    async for token, statee in rwkv_inference_tokens(fullprompt_tokens, model, pipeline, typicalSampling=typicalSampling, state=statee):
        full_response += token
        response += token
        token_count += 1
        yield token
        await asyncio.sleep(0.000001)
    
    # Save the total tokens
    input_token_count = len(fullprompt_tokens)
    INFERENCE_PROMPT_TOKENS += input_token_count
    INFERENCE_OUTPUT_TOKENS += token_count

    print (f"## Prompt ({input_token_count} tokens) ##")
    print (fullprompt)
    print (f"## Response ({token_count} tokens) ##")
    print (response)
    print ("##################")
        
    # cacheKey = full_response.strip() + "<|im_end|>\n"
    # statee = [state.cpu() for state in statee]
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
    global CONCURRENT_REQ_COUNT, CONCURRENT_REQ_LIMIT, INFERENCE_TIME_TAKEN_S

    print("[CHAT] request started", request.path)

    # Concurrency locking
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
        
        time_taken_s = time.time() - startTime
        INFERENCE_TIME_TAKEN_S += time_taken_s

        print(f"## Time taken: {time_taken_s} ##")
        # print(f"## Tokens generated: {totalTokens} ##")
        # print(f"## Tokens per second: {totalTokens / (time.time() - startTime)} ##")
        
        await response.write_eof()
        return response
    except OSError:
        print("## Client disconnected ##")
        # unlockModel(index)
    finally:
        CONCURRENT_REQ_COUNT += -1

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
    global INFERENCE_TIME_TAKEN_S, INFERENCE_PROMPT_TOKENS, INFERENCE_OUTPUT_TOKENS

    TOTAL_TOKENS_COUNT = INFERENCE_PROMPT_TOKENS + INFERENCE_OUTPUT_TOKENS
    while True:
        print(
            f"\n~~ Concurrent req: {CONCURRENT_REQ_COUNT}",
            f"\n~~ cummulative token count: {TOTAL_TOKENS_COUNT} tokens"
            f"\n~~ cummulative inference time: {INFERENCE_TIME_TAKEN_S}s"
            f"\n~~ tokens per second: {TOTAL_TOKENS_COUNT/INFERENCE_TIME_TAKEN_S}"
        )
        await asyncio.sleep(5)
    
def background_starter():
    asyncio.run(background_process())

threading.Thread(target=background_starter).start()

# THIS IS BLOCKING
web.run_app(app, port=9997)