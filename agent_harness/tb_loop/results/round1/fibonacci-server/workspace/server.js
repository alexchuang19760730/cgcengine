const http = require('http');
const { URL } = require('url');

function fib(n) {
  if (n === 0) return 0;
  if (n === 1) return 1;
  let a = 0n, b = 1n;
  for (let i = 2; i <= n; i++) {
    [a, b] = [b, a + b];
  }
  return b;
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://localhost:${req.headers.host || 3000}`);
  if (url.pathname !== '/fib') {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Not Found' }));
    return;
  }
  const nParam = url.searchParams.get('n');
  if (nParam === null || nParam === '') {
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Bad Request' }));
    return;
  }
  if (!/^-?\d+$/.test(nParam)) {
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Bad Request' }));
    return;
  }
  const n = Number(nParam);
  if (!Number.isSafeInteger(n)) {
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Bad Request' }));
    return;
  }
  if (n < 0) {
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Bad Request' }));
    return;
  }
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ result: fib(n).toString() }));
});

server.listen(3000, () => {
  console.log('Server listening on port 3000');
});
