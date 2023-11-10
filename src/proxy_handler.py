
from aiohttp import web
import logging
import aiohttp
from urllib.parse import urlsplit

# Base URL for openAI
base_url = "https://api.openai.com"

# Proxy handling
async def proxy_handler(request):
    print("[PROXY] request started", request.path)

    url = base_url + request.path
    method = request.method
    headers = dict(request.headers)
    headers['Host'] = "{0.netloc}".format(urlsplit(base_url))
    
    req_text = await request.text()
    
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