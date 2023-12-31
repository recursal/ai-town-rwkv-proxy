import copy
import os, gc, torch
import time
import logging
import sys

from pynvml import *
from torch.nn import functional as F
import numpy as np
import json
from rwkv_tokenizer import TRIE_TOKENIZER
from sample_logits import sample_logits

from urllib.parse import urlsplit
import random

from aiohttp import web
import concurrent
import asyncio

# nvmlInit()
# gpu_h = nvmlDeviceGetHandleByIndex(0)
ctx_limit = 8192
ctx_gpt_mode_chunks = 1024

#os.environ["CUDA_VISIBLE_DEVICES"] = ''
os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '0' # if '1' then use CUDA kernel for seq mode (much faster)

global CONCURRENT_REQ_COUNT, CONCURRENT_REQ_LIMIT
CONCURRENT_REQ_COUNT = 0
CONCURRENT_REQ_LIMIT = 50

global INFERENCE_TIME_TAKEN_S, TOTAL_PROMPT_TOKENS, TOTAL_OUTPUT_TOKENS
INFERENCE_TIME_TAKEN_S = 1
TOTAL_PROMPT_TOKENS = 1
TOTAL_OUTPUT_TOKENS = 1

torch.set_num_threads(14)

# Get the filename, and system strat
model_filename = "rwkv-1b5-ai-town-v1.3.pth" if len(sys.argv) <= 1 else sys.argv[1]
model_strategy = "cpu fp32" if len(sys.argv) <= 2 else sys.argv[2]
if model_filename == None:
    model_filename = "rwkv-1b5-ai-town-v1.3.pth"
if model_strategy == None:
    model_strategy = "cpu fp32"

# RWKV pip libraries
from proxy_handler import proxy_handler
from rwkv_inference import rwkv_inference_tokens

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

current_dir = os.path.dirname( os.path.dirname(os.path.realpath(__file__)) )
model_path = current_dir + f"/{model_filename}"

models = [
    RWKV(model=model_path, strategy=model_strategy)
]
pipelines = []
for model in models:
    pipelines.append(TRIE_TOKENIZER(current_dir + "/rwkv_vocab_v20230922_chatml.txt"))

# set thread count with pytorch

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
    global TOTAL_PROMPT_TOKENS, TOTAL_OUTPUT_TOKENS

    typicalSampling = True
    
    fullprompt = await buildPrompt(conversation)
    fullprompt_tokens = pipeline.encode(fullprompt)

    statee = None

    full_response = fullprompt
    response = ""

    output_token_count = 0
    async for token, statee in rwkv_inference_tokens(fullprompt_tokens, model, pipeline, typicalSampling=typicalSampling, state=statee):
        full_response += token
        response += token
        output_token_count += 1
        yield token
        await asyncio.sleep(0.000001)
    
    # Save the total tokens
    input_token_count = len(fullprompt_tokens)
    TOTAL_PROMPT_TOKENS += input_token_count
    TOTAL_OUTPUT_TOKENS += output_token_count

    print(f">> DEBUG: TOTAL_PROMPT_TOKENS {TOTAL_PROMPT_TOKENS}")
    print(f">> DEBUG: TOTAL_OUTPUT_TOKENS {TOTAL_OUTPUT_TOKENS}")

    print (f"## Prompt ({input_token_count} tokens) ##")
    print (fullprompt)
    print (f"## Response ({output_token_count} tokens) ##")
    print (response)
    print ("##################")
        
    # cacheKey = full_response.strip() + "<|im_end|>\n"
    # statee = [state.cpu() for state in statee]
    # cachedStates[hash(cacheKey)] = (statee, time.time() + 60 * 60) # cache state for 1 hour
    # gc.collect()

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

    # Start timestamp
    startTime = time.time()

    # Try block
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

#
# I COULD NOT GET THIS TO WORK
# SOMEONE WITH MORE EXPERIENCE WITH SERVER PYTHON, PLEASE FIX IT
# 
async def chat_handle_fork(request):
    # try:
    #     # This does not work, throws error "printHelloWorld Needs to be awaited"
    #     thread = threading.Thread(target=chat_handle, args=(request,))
    #     thread.start()
    # except (KeyboardInterrupt, SystemExit):
    #     # Stop Thread when CTRL + C is pressed or when program is exited
    #     thread.join()

    # 2. Run in a custom thread pool:
    loop = asyncio.get_event_loop()
    
    response = None
    with concurrent.futures.ThreadPoolExecutor() as executor:
        loop_response = await loop.run_in_executor(
            executor,
            lambda: asyncio.run_coroutine_threadsafe(chat_handle(request), loop)
        )
        response = await loop_response.result()
    return response

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
    global CONCURRENT_REQ_COUNT, CONCURRENT_REQ_LIMIT, INFERENCE_TIME_TAKEN_S, TOTAL_PROMPT_TOKENS, TOTAL_OUTPUT_TOKENS

    while True:
        total_token_count = TOTAL_PROMPT_TOKENS + TOTAL_OUTPUT_TOKENS
        print(
            f"\n~~ Concurrent req: {CONCURRENT_REQ_COUNT}",
            f"\n~~ cummulative input/output count: {TOTAL_PROMPT_TOKENS} / {TOTAL_OUTPUT_TOKENS} tokens",
            f"\n~~ cummulative inference time: {INFERENCE_TIME_TAKEN_S}s",
            f"\n~~ tokens per second: {total_token_count/INFERENCE_TIME_TAKEN_S}"
        )
        await asyncio.sleep(5)
    
def background_starter():
    asyncio.run(background_process())

threading.Thread(target=background_starter).start()

# THIS IS BLOCKING
web.run_app(app, port=9997)
