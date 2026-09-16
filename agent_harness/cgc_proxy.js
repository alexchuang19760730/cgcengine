#!/usr/bin/env node
const http = require("http");
const url = require("url");
const crypto = require("crypto");
const CGC = process.env.CGC_URL || "http://192.168.101.87:1237";
const MODEL = "qwen3.6-35b-mtp";

function buildPrompt(msgs, sysTxt) {
  const p = [];
  if (sysTxt) {
    const t = Array.isArray(sysTxt) ? sysTxt.map(b => b.text || "").join("
") : String(sysTxt);
    p.push("[system]
" + t);
  }
  for (const m of msgs) {
    let c = m.content || "";
    if (Array.isArray(c)) c = c.filter(b => b.type === "text").map(b => b.text).join("
");
    p.push("[" + (m.role || "user") + "]
" + String(c));
  }
  return p.join("

");
}

function se(evt, data) {
  return "event: " + evt + "
data: " + JSON.stringify(data) + "

";
}

const server = http.createServer((req, res) => {
  const u = url.parse(req.url);
  if (req.method === "GET" && u.pathname === "/v1/models") {
    res.writeHead(200, {"Content-Type":"application/json"});
    return res.end(JSON.stringify({data:[{id:MODEL}]}));
  }
  if (req.method === "POST" && u.pathname === "/v1/messages") {
    let body = "";
    req.on("data", d => body += d);
    req.on("end", () => {
      let p; try { p = JSON.parse(body); } catch(e) { res.writeHead(400); return res.end("{}"); }
      const prompt = buildPrompt(p.messages||[], p.system);
      const mx = p.max_tokens || 4096;
      const stream = p.stream || false;
      console.log("[proxy] len=" + prompt.length + " stream=" + stream);
      const pd = JSON.stringify({prompt, max_tokens: mx, seed: 42});
      const opts = {hostname:"192.168.101.87", port:1237, path:"/v1/cgc/resume", method:"POST",
        headers:{"Content-Type":"application/json","Content-Length":Buffer.byteLength(pd)}};
      const pr = http.request(opts, cr => {
        if (!stream) {
          let txt="", buf="";
          cr.on("data", ch => {
            buf += ch.toString(); const ls = buf.split("
"); buf = ls.pop();
            for (const l of ls) { if (!l.startsWith("data: ")) continue;
              try { const ev = JSON.parse(l.slice(6)); if (ev.event==="token") txt += ev.t||""; } catch(e){}
            }
          });
          cr.on("end", () => {
            res.writeHead(200, {"Content-Type":"application/json"});
            res.end(JSON.stringify({id:"msg_"+crypto.randomBytes(12).toString("hex"),
              type:"message",role:"assistant",content:[{type:"text",text:txt}],
              model:MODEL,stop_reason:"end_turn",usage:{input_tokens:0,output_tokens:txt.split(/s+/).length}}));
          });
        } else {
          res.writeHead(200, {"Content-Type":"text/event-stream","Cache-Control":"no-cache","Connection":"keep-alive"});
          const mid = "msg_"+crypto.randomBytes(12).toString("hex");
          res.write(se("message_start",{type:"message",id:mid,role:"assistant",content:[],model:MODEL,stop_reason:null,usage:{input_tokens:0,output_tokens:0}}));
          res.write(se("content_block_start",{type:"content_block_start",index:0,content_block:{type:"text",text:""}}));
          let buf="";
          cr.on("data", ch => {
            buf += ch.toString(); const ls = buf.split("
"); buf = ls.pop();
            for (const l of ls) { if (!l.startsWith("data: ")) continue;
              try { const ev = JSON.parse(l.slice(6));
                if (ev.event==="token") res.write(se("content_block_delta",{type:"content_block_delta",index:0,delta:{type:"text_delta",text:ev.t||""}}));
                else if (ev.event==="summary") console.log("[proxy] done:",ev.n_decoded,"tok",ev.decode_tps?.toFixed(1),"t/s");
              } catch(e){}
            }
          });
          cr.on("end", () => {
            res.write(se("content_block_stop",{type:"content_block_stop",index:0}));
            res.write(se("message_delta",{type:"message_delta",delta:{stop_reason:"end_turn"},usage:{output_tokens:0}}));
            res.write(se("message_stop",{type:"message_stop"}));
            res.end();
          });
        }
      });
      pr.on("error", e => { res.writeHead(502); res.end(JSON.stringify({error:e.message})); });
      pr.write(pd); pr.end();
    });
    return;
  }
  res.writeHead(404); res.end();
});

const port = parseInt(process.argv[2]) || 8082;
server.listen(port, "127.0.0.1", () => {
  console.log("[proxy] Anthropic->CGC on :" + port + " -> " + CGC);
  console.log("[proxy] ANTHROPIC_BASE_URL=http://127.0.0.1:" + port + " ANTHROPIC_API_KEY=dummy claude");
});