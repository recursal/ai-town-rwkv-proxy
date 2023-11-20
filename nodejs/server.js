const http = require('http');
const url = require('url');

// Utility script to read chunks from a stream
function readChunks(reader) {
    return {
        async* [Symbol.asyncIterator]() {
            let readResult = await reader.read();
            while (!readResult.done) {
                yield readResult.value;
                readResult = await reader.read();
            }
        },
    };
}

// Create an HTTP server
const server = http.createServer((req, res) => {
    let body = [];
    let pathname = url.parse(req.url).pathname;

    // Log the request
    console.log('## Handling request at: ' + pathname);

    // Collect the request data
    req.on('data', chunk => {
        body.push(chunk);
    }).on('end', async () => {
        body = Buffer.concat(body).toString();
        console.log('## Request payload: \n' + body);

        // Forward to different ports based on the URL
        let targetUrl;
        if (pathname.startsWith('/v1/chat/completions')) {
            targetUrl = 'http://localhost:9995/api/oai' + pathname;
        } else if (pathname.startsWith('/v1/embeddings')) {
            targetUrl = 'http://localhost:9991/api/oai' + pathname;
        } else {
            res.writeHead(404);
            res.end();
            return;
        }

        // Forward the request
        const response = await fetch(targetUrl, {
            method: req.method,
            headers: req.headers,
            body: body
        });

        // Forward the response
        res.writeHead(response.status, response.headers);
		
		// Build the full output
		// fullOutput = []

		// Get the reader
		const reader = response.body.getReader();
        for await (const chunk of readChunks(reader)) {
            res.write(chunk);
			// fullOutput.push(chunk)
        }

		// End the stream
		res.end();
    });
});

// Log the server start
server.on('listening', () => {
    console.log('## RWKV proxy Server listening on port 9997');
});

// Start the server
server.listen(9997);