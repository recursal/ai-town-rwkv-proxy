# AI Town - RWKV Proxy

> Disclaimer: Currently it is not possible to run AI town fully offline, as it requires
> - convex for the api backend + vector DB
> - openAI embeddings

However this project is to help put that a step closer, in running a full AI town locally on any modern device

# Setup steps

## Step 1 - Setup AI town as per normal

See: https://github.com/a16z-infra/ai-town
Make sure it works first, before rerouting to RWKV

## Step 2 - Clone the AI-TOWN-RWKV-proxy project

```
git clone https://github.com/recursal/ai-town-rwkv-proxy.git
cd ai-town-rwkv-proxy

./setup-and-run.sh
```

For subsequent runs you can just do just

```
./run.sh
```

This will setup an API proxy (for embedding) + rwkv (for chat completion) at port 9997

## Step 3 - Deploy the AI Town proxy via cloudflared

Due to current limitations, you will need to route your RWKV AI model, through a public URL. There are multiple ways to do it, but the easiest and most reliable is cloudflared which you can install with

```bash
npm install -g cloudflared

####
# PS: cloudflared was built for x86, if you are running on ARM based macs, you may need to get rosette installed
####
# softwareupdate --install-rosetta
# softwareupdate --install-rosetta --agree-to-license
```

After installing, you can get a public URL with just

```bash
# Create a public URL, pointing to port 9997 (which is what we run our API on for now)
cloudflared --url http://localhost:9997
```

This will give an output like the following

![Cloudflared URL example](./guides/img/cloudflared-url.png)

## Step 4 - Route the convex OpenAI request to the proxy

Under the convex environment settings, add the OPENAI_API_BASE (do not include the ending slash)

![Convex environment settings](./guides/img/convex_env.png)