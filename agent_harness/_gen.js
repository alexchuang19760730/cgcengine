const fs = require("fs");
const NL = String.fromCharCode(10);
const lines = [];
function w(s) { lines.push(s); }
w("#!/usr/bin/env node");
w("const http = require(\"http\");");
w("const url = require(\"url\");");
w("const crypto = require(\"crypto\");");
w("const CGC = process.env.CGC_URL || \"http://192.168.101.87:1237\";");
w("const MODEL = \"qwen3.6-35b-mtp\";");
