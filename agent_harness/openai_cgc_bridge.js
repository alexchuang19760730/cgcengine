const http = require("http");
const crypto = require("crypto");
const CGC = "http://192.168.101.87:1237";

function buildPrompt(msgs) {
  return msgs.map(m => {
    let c = m.content || "";
    if (Array.isArray(c)) c = c.filter(b=>b.type==="text").map(b=>b.text).join("
");
    return "[" + (m.role||"user") + "]
" + c;
  }).join("

");
}

http.createServer((req, res) => {
  if (req.method === "GET" && req.url === "/v1/models") {
    res.writeHead(200, {"Content-Type":"application/json"});
    return res.end(JSON.stringify({data:[{id:"qwen3.6-35b-mtp"}]}));
  }
  if (req.method === "POST" && req.url === "/v1/chat/completions") {
    let body = "";
    req.on("data", d => body += d);
    req.on("end", () => {
      const p = JSON.parse(body);
      const prompt = buildPrompt(p.messages || []);
      const mx = p.max_tokens || 4096;
      const stream = p.stream || false;
      const pd = JSON.stringify({prompt, max_tokens: mx, seed: 42});
      const pr = http.request({hostname:"192.168.101.87",port:1237,path:"/v1/cgc/resume",method:"POST",
        headers:{"Content-Type":"application/json","Content-Length":Buffer.byteLength(pd)}}, cr => {
        if (!stream) {
          let txt="",buf="";
          cr.on("data", ch => { buf+=ch.toString(); const ls=buf.split("
");buf=ls.pop();
            for(const l of ls){if(!l.startsWith("data: "))continue;try{const ev=JSON.parse(l.slice(6));
              if(ev.event==="token")txt+=ev.t||"";}catch(e){}}
          });
          cr.on("end",()=>{
            res.writeHead(200,{"Content-Type":"application/json"});
            res.end(JSON.stringify({id:"chatcmpl-"+crypto.randomBytes(8).toString("hex"),
              object:"chat.completion",model:"qwen3.6-35b-mtp",
              choices:[{index:0,message:{role:"assistant",content:txt},finish_reason:"stop"}],
              usage:{prompt_tokens:0,completion_tokens:txt.split(/s+/).length,total_tokens:0}}));
          });
        } else {
          res.writeHead(200,{"Content-Type":"text/event-stream","Cache-Control":"no-cache"});
          let buf="";
          cr.on("data",ch=>{
            buf+=ch.toString();const ls=buf.split("
");buf=ls.pop();
            for(const l of ls){if(!l.startsWith("data: "))continue;
              try{const ev=JSON.parse(l.slice(6));
                if(ev.event==="token")res.write("data: "+JSON.stringify({id:"chatcmpl-x",object:"chat.completion.chunk",
                  model:"qwen3.6-35b-mtp",choices:[{index:0,delta:{content:ev.t||""},finish_reason:null}]})+"

");
              }catch(e){}}
          });
          cr.on("end",()=>{res.write("data: [DONE]

");res.end();});
        }
      });
      pr.on("error",e=>{res.writeHead(502);res.end(JSON.stringify({error:e.message}));});
      pr.write(pd);pr.end();
    });
    return;
  }
  res.writeHead(404);res.end();
}).listen(8083, "127.0.0.1", ()=>console.log("[bridge] OpenAI->CGC on :8083 -> "+CGC));