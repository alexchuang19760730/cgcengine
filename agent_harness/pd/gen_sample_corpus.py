import os, sys
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
#!/usr/bin/env python3
"""鐢熸垚澶氭ǎ鍖?sample corpus, 鐢ㄦ柤 MoT-h 瑷撶反灏嶆帯闆?

鐢熸垚 5 椤炴枃鏈? 姣忛 N 姊? 瀵叆 corpus/ 璩囨枡澶?
瑕嗚搵: 浠ｇ⒓ / 灏嶈┍ / 涓嫳娣?/ 闀锋枃 / 鏁稿鎺ㄧ悊.

鐢ㄦ硶:
  py gen_sample_corpus.py --output corpus/ --per-category 20

杓稿嚭:
  corpus/
    鈹溾攢鈹€ code_001.py
    鈹溾攢鈹€ code_002.py
    ...
    鈹溾攢鈹€ dialog_001.txt
    ...
    鈹溾攢鈹€ mixed_001.txt
    ...
    鈹溾攢鈹€ long_001.md
    ...
    鈹斺攢鈹€ math_001.txt
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------------------------
# 浠ｇ⒓妯ｆ湰 (Python/Swift/JS)
# ---------------------------------------------------------------------------
CODE_TEMPLATES = [
    """# Fibonacci 鏁稿垪 - 杩唬鐗?
def fibonacci(n: int) -> list[int]:
    \"\"\"杩斿洖鍓?n 鍊?Fibonacci 鏁?\"\"\"
    if n <= 0:
        return []
    if n == 1:
        return [0]
    fibs = [0, 1]
    for i in range(2, n):
        fibs.append(fibs[-1] + fibs[-2])
    return fibs


if __name__ == "__main__":
    print(fibonacci(10))
    # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
""",

    """# 浜屽垎鎼滅储
def binary_search(arr: list[int], target: int) -> int:
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


arr = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
print(binary_search(arr, 7))   # 3
print(binary_search(arr, 10))  # -1
""",

    """// JavaScript: 绨″柈鐨?debounce 鍑芥暩
function debounce(fn, delay) {
  let timer = null;
  return function(...args) {
    clearTimeout(timer);
    timer = setTimeout(() => {
      fn.apply(this, args);
    }, delay);
  };
}

const log = debounce((msg) => console.log(msg), 300);
log("hello");
log("world");  // 鍙渻鍗?"world"
""",

    """// Swift: 绨″柈鐨?Stack 绲愭
struct Stack<T> {
    private var items: [T] = []

    var isEmpty: Bool { items.isEmpty }
    var count: Int { items.count }

    mutating func push(_ item: T) {
        items.append(item)
    }

    mutating func pop() -> T? {
        return items.popLast()
    }

    func peek() -> T? {
        return items.last
    }
}

var stack = Stack<Int>()
stack.push(1)
stack.push(2)
print(stack.pop() ?? "empty")  // 2
""",

    """# Python: 绨″柈鐨?LRU Cache
from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value


cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
print(cache.get(1))  # 1
cache.put(3, 3)      # evict key 2
print(cache.get(2))  # -1
""",

    """# Python: 蹇€熸帓搴?
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


print(quicksort([3, 6, 1, 8, 2, 9, 4]))
# [1, 2, 3, 4, 6, 8, 9]
""",

    """# Python: 绨″柈 HTTP server
from http.server import HTTPServer, BaseHTTPRequestHandler
import json


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "path": self.path,
            "method": "GET"
        }).encode())


if __name__ == "__main__":
    server = HTTPServer(("localhost", 8000), Handler)
    print("Serving on http://localhost:8000")
    server.serve_forever()
""",

    """// TypeScript: 绨″柈鐨?Observable
interface Observer<T> {
  next: (value: T) => void;
  error: (err: any) => void;
  complete: () => void;
}

class Observable<T> {
  constructor(private subscribeFn: (obs: Observer<T>) => () => void) {}

  subscribe(obs: Observer<T>): () => void {
    return this.subscribeFn(obs);
  }
}

const counter = new Observable<number>((obs) => {
  let n = 0;
  const id = setInterval(() => {
    obs.next(n++);
    if (n >= 5) {
      obs.complete();
      clearInterval(id);
    }
  }, 100);
  return () => clearInterval(id);
});

counter.subscribe({
  next: (v) => console.log("got", v),
  error: (e) => console.error(e),
  complete: () => console.log("done"),
});
""",
]


# ---------------------------------------------------------------------------
# 灏嶈┍妯ｆ湰
# ---------------------------------------------------------------------------
DIALOG_TEMPLATES = [
    """User: 浣犲ソ, 璜嬪晱 Python 鐨?list 鍜?tuple 鏈変粈楹煎崁鍒?
Assistant: 涓昏鍗€鍒ユ湁鍏╅粸:
1. list 鏄彲璁婄殑, tuple 鏄笉鍙畩鐨?
2. tuple 閫氬父鐢ㄦ柤鐣版鏁告摎 (濡傚骇妯?, list 鐢ㄦ柤鍚屾鏁告摎

渚嬪:
  coords = (3, 4)       # tuple, 涓嶆渻璁?
  points = [1, 2, 3]    # list, 鍙互 append/pop

User: 閭ｆ€ц兘涓婃湁宸垾鍡?
Assistant: tuple 鍥犵偤涓嶅彲璁? 鍓靛缓閫熷害绋嶅揩, 鍏у瓨浣旂敤鐣ュ皬. 浣嗗湪澶ч儴鍒嗗牬鏅笅宸暟涓嶉’钁?""",

    """User: 瑙ｉ噵涓€涓嬩粈楹兼槸闁夊寘 (closure).
Assistant: 闁夊寘鏄嚱鏁?+ 鍏惰娉曠挵澧冪殑绲勫悎. 鍏у堡鍑芥暩鍙互瑷晱澶栧堡鍑芥暩鐨勮畩閲? 鍗充娇澶栧堡鍑芥暩宸茬稉杩斿洖.

```python
def make_counter():
    count = 0
    def inc():
        nonlocal count
        count += 1
        return count
    return inc

c = make_counter()
print(c(), c(), c())  # 1 2 3
```

閫欒！ `inc` 闁夊寘浜?`count` 璁婇噺.

User: 鐐轰粈楹艰 nonlocal?
Assistant: 鍥犵偤 Python 榛樿獚鎶婅肠鍊肩暥浣滃壍寤哄眬閮ㄨ畩閲? nonlocal 鍛婅ù Python 閫欏€嬭畩閲忓湪澶栧堡浣滅敤鍩?""",

    """User: What is the time complexity of binary search?
Assistant: O(log n) where n is the array length. Each comparison halves the search space.

For example, searching a sorted array of 1 million elements takes at most ~20 comparisons (log2(10^6) 鈮?20).

User: What about space complexity?
Assistant: O(1) for the iterative version, O(log n) for the recursive version (call stack).""",

    """User: 瑙ｉ噵涓€涓?RESTful API.
Assistant: REST 鏄竴绋?API 瑷▓棰ㄦ牸, 鏍稿績鍘熷墖:
- 鐢?HTTP method 琛ㄩ仈鎿嶄綔: GET (鏌?, POST (寤?, PUT (鏀?, DELETE (鍒?
- 鐢?URL 琛ㄩ仈璩囨簮: /users/123
- 鐒＄媭鎱? 姣忓€嬭珛姹傝嚜鍖呭惈

渚嬪:
  GET    /users        # 鍒楀嚭鎵€鏈夌敤鎴?
  GET    /users/123    # 鍙栧緱鍠€嬬敤鎴?
  POST   /users        # 鍓靛缓鐢ㄦ埗
  PUT    /users/123    # 鏇存柊鐢ㄦ埗
  DELETE /users/123    # 鍒櫎鐢ㄦ埗

User: 閭?PATCH 鍛?
Assistant: PATCH 鐢ㄦ柤閮ㄥ垎鏇存柊, 鍙櫦閫佽畩鍖栫殑瀛楁. PUT 鏄叏閲忔浛鎻?""",
]


# ---------------------------------------------------------------------------
# 涓嫳娣峰悎闀锋枃
# ---------------------------------------------------------------------------
MIXED_TEMPLATES = [
    """# 姗熷櫒瀛哥繏鍩虹

Machine learning is a subset of artificial intelligence that enables systems to learn from data.

## 鐩ｇ潱寮忓缈?(Supervised Learning)

鐩ｇ潱寮忓缈掍娇鐢ㄦ瑷樻暩鎿氳〒绶存ā鍨? 甯歌浠诲嫏鍖呮嫭:
- Classification: 鍒嗛鍟忛, 濡傚湒鐗囪鲸璀?
- Regression: 鍥炴鍟忛, 濡傛埧鍍归爯娓?

渚嬪, 绲﹀畾涓€鎵硅矒鐙楀湒鐗囧拰妯欑堡, 瑷撶反涓€鍊嬪垎椤炲櫒:
```python
from sklearn.svm import SVC
clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

## 闈炵洠鐫ｅ紡瀛哥繏 (Unsupervised Learning)

闈炵洠鐫ｅ紡瀛哥繏铏曠悊鐒℃绫ゆ暩鎿? 甯歌鏂规硶:
- Clustering: 鑱氶, 濡?K-means
- Dimensionality reduction: 闄嶇董, 濡?PCA

## 娣卞害瀛哥繏 (Deep Learning)

娣卞害瀛哥繏浣跨敤澶氬堡绁炵稉缍茬怠. 涓昏椤炲瀷:
1. CNN (Convolutional Neural Network) - 鍦栧儚
2. RNN (Recurrent Neural Network) - 搴忓垪
3. Transformer - 瑾炶█妯″瀷 (濡?GPT, BERT)

Transformer 鐨勬牳蹇冩槸 self-attention 姗熷埗, 璁撴ā鍨嬭兘闂滄敞搴忓垪涓换鎰忎綅缃殑 token.""",

    """# 瑷堢畻姗熺恫璺熀绀?

Computer networks connect devices to share resources and communicate.

## OSI 涓冨堡妯″瀷

The OSI model has 7 layers:
1. Physical (鐗╃悊灞? - 浣嶅厓鍌宠几
2. Data Link (璩囨枡閺堢祼灞? - 骞€鍌宠几, MAC 瀹氬潃
3. Network (缍茶矾灞? - IP 璺敱
4. Transport (鍌宠几灞? - TCP/UDP, 绔彛
5. Session (鏈冭灞? - 鏈冭┍绠＄悊
6. Presentation (琛ㄩ仈灞? - 鍔犲瘑, 澹撶府
7. Application (鎳夌敤灞? - HTTP, FTP, SMTP

## TCP vs UDP

TCP (Transmission Control Protocol):
- 鍙潬鍌宠几, 涓夋鎻℃墜寤虹珛閫ｆ帴
- 鏈夊簭, 鐒′笩鍖?
- 鎳夌敤: HTTP, SMTP, FTP

UDP (User Datagram Protocol):
- 涓嶅彲闈? 鐒￠€ｆ帴
- 閫熷害蹇? 鍙笩鍖?
- 鎳夌敤: DNS, 瑕栭牷娴? 閬婃埐

## HTTP 鍗旇

HTTP 鏄劇鐙€鎱嬬殑 request-response 鍗旇. 甯歌鐙€鎱嬬⒓:
- 200 OK
- 301 Moved Permanently
- 404 Not Found
- 500 Internal Server Error""",

    """# 璩囨枡绲愭鑸囨紨绠楁硶

Data structures organize data for efficient access. Algorithms solve problems step by step.

## 鍩烘湰璩囨枡绲愭

### Array (闄ｅ垪)
- 閫ｇ簩鍏у瓨, O(1) 闅ㄦ瀛樺彇
- 鎻掑叆/鍒櫎 O(n)

### Linked List (閺堢祼涓插垪)
- 闈為€ｇ簩, 閫氶亷鎸囬嚌閫ｆ帴
- 鎻掑叆/鍒櫎 O(1) (宸茬煡绡€榛?
- 瀛樺彇 O(n)

### Hash Table (闆滄箠琛?
- Key-value 鏄犲皠
- 骞冲潎 O(1) 鏌ユ壘/鎻掑叆/鍒櫎
- Python dict, JavaScript Map

### Tree (妯?
- 闅庡堡绲愭
- 浜屽厓鎼滅储妯? 宸?< 鏍?< 鍙?
- 骞宠　妯? AVL, Red-Black

## 鎺掑簭婕旂畻娉?

| Algorithm | Time (avg) | Time (worst) | Space | Stable |
|-----------|------------|--------------|-------|--------|
| Quicksort | O(n log n) | O(n虏)        | O(log n) | No  |
| Mergesort | O(n log n) | O(n log n)   | O(n)  | Yes    |
| Heapsort  | O(n log n) | O(n log n)   | O(1)  | No     |
| Bubble    | O(n虏)      | O(n虏)        | O(1)  | Yes    |

## 鎼滃皨婕旂畻娉?

- Linear search: O(n)
- Binary search: O(log n) (闇€鎺掑簭)
- Hash lookup: O(1) 骞冲潎""",
]


# ---------------------------------------------------------------------------
# 闀锋枃妯ｆ湰
# ---------------------------------------------------------------------------
LONG_TEMPLATES = [
    """# 杌熼珨宸ョ▼鍘熷墖

Software engineering is the systematic application of engineering principles to software.

## 1. 鍠竴鑱疯铂鍘熷墖 (Single Responsibility Principle)

涓€鍊嬮鍒ユ垨妯＄祫鎳夎┎鍙湁涓€鍊嬭伔璨? 閫欐剰鍛宠憲瀹冨彧鏈変竴鍊嬭畩鍖栫殑鍘熷洜.

Bad example:
```python
class User:
    def save_to_db(self): ...
    def send_email(self): ...
    def generate_report(self): ...
```

Good:
```python
class User: pass
class UserRepository:
    def save(self, user): ...
class EmailService:
    def send(self, user, msg): ...
class ReportGenerator:
    def generate(self, user): ...
```

## 2. 闁嬫斁灏侀枆鍘熷墖 (Open/Closed Principle)

杌熼珨瀵﹂珨鎳夎┎灏嶆摯灞曢枊鏀? 灏嶄慨鏀瑰皝闁? 閫氶亷绻兼壙鎴栫祫鍚堟坊鍔犳柊鍔熻兘, 鑰屼笉鏄慨鏀圭従鏈変唬纰?

## 3. 閲屾皬鏇挎彌鍘熷墖 (Liskov Substitution Principle)

瀛愰鍒ュ繀闋堣兘鏇挎彌鐖堕鍒ヨ€屼笉鐮村绋嬪紡琛岀偤. 瀛愰鍒ヤ笉鎳夎┎鍔犳洿鍤存牸鐨勫墠缃浠? 鎴栨洿寮辩殑寰岀疆姊濅欢.

## 4. 浠嬮潰闅旈洟鍘熷墖 (Interface Segregation Principle)

瀹㈡埗绔笉鎳夎┎琚揩渚濊炒瀹冧笉浣跨敤鐨勬柟娉? 澶氬€嬪皥鐢ㄤ粙闈㈠劒鏂间竴鍊嬮€氱敤浠嬮潰.

## 5. 渚濊炒鍙嶈綁鍘熷墖 (Dependency Inversion Principle)

楂樺堡妯＄祫涓嶆噳瑭蹭緷璩翠綆灞ゆā绲? 鍏╄€呴兘鎳夎┎渚濊炒鎶借薄. 鎶借薄涓嶆噳瑭蹭緷璩寸窗绡€, 绱扮瘈鎳夎┎渚濊炒鎶借薄.

## 瑷▓妯″紡

### 鍓靛缓鍨嬫ā寮?
- Singleton: 纰轰繚涓€鍊嬮鍙湁涓€鍊嬪渚?
- Factory Method: 瀹氱京鍓靛缓鐗╀欢鐨勪粙闈? 璁撳瓙椤炴焙瀹氬渚嬪寲鍝€嬮
- Abstract Factory: 鍓靛缓涓€绯诲垪鐩搁棞鐗╀欢
- Builder: 鍒嗘椹熸寤鸿闆滅墿浠?

### 绲愭鍨嬫ā寮?
- Adapter: 杞夋彌浠嬮潰
- Decorator: 鍕曟厠娣诲姞鑱疯铂
- Facade: 鎻愪緵绲变竴浠嬮潰
- Proxy: 鎺у埗瀛樺彇

### 琛岀偤鍨嬫ā寮?
- Observer: 鐧煎竷-瑷傞柋
- Strategy: 灏佽鍙簰鎻涚殑婕旂畻娉?
- Command: 灏囪珛姹傚皝瑁濈偤鐗╀欢
- Iterator: 闋嗗簭瀛樺彇鑱氬悎鍏冪礌

## 娓│椹呭嫊闁嬬櫦 (TDD)

TDD 娴佺▼: Red 鈫?Green 鈫?Refactor
1. Red: 瀵竴鍊嬪け鏁楃殑娓│
2. Green: 瀵渶灏戜唬纰艰畵娓│閫氶亷
3. Refactor: 鏀归€蹭唬纰肩祼妲?

鍎粸:
- 鏇存竻鏅扮殑浠嬮潰瑷▓
- 鍗虫檪鍙嶉
- 閲嶆淇″績
- 鏂囨獢鍖栭爯鏈熻鐐?"",

    """# 鐝句唬 Web 闁嬬櫦

Modern web development spans frontend, backend, and infrastructure.

## Frontend

### HTML/CSS/JavaScript 鍩虹

HTML 绲愭鍖栧収瀹? CSS 鎺у埗妯ｅ紡, JavaScript 娣诲姞浜掑嫊.

```html
<!DOCTYPE html>
<html>
<head>
  <title>My App</title>
  <style>
    body { font-family: sans-serif; }
    .container { max-width: 800px; margin: 0 auto; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Hello</h1>
    <button onclick="alert('clicked')">Click</button>
  </div>
</body>
</html>
```

### React

React 鏄伈鏄庡紡 UI 搴? 浣跨敤绲勪欢鍜岃櫅鎿?DOM.

```jsx
function Counter() {
  const [count, setCount] = useState(0);
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>+1</button>
    </div>
  );
}
```

Hooks:
- useState: 鐙€鎱嬬鐞?
- useEffect: 鍓綔鐢?
- useContext: 鍏ㄥ煙鐙€鎱?
- useMemo/useCallback: 鎬ц兘鍎寲

### Vue

Vue 鏄几閫插紡妗嗘灦, 妯℃澘瑾炴硶鏇存帴杩?HTML.

```vue
<template>
  <div>
    <p>{{ count }}</p>
    <button @click="count++">+1</button>
  </div>
</template>

<script setup>
import { ref } from 'vue';
const count = ref(0);
</script>
```

## Backend

### Node.js + Express

```javascript
const express = require('express');
const app = express();

app.get('/api/users', (req, res) => {
  res.json([{ id: 1, name: 'Alice' }]);
});

app.listen(3000);
```

### Python + FastAPI

```python
from fastapi import FastAPI

app = FastAPI()

@app.get('/api/users')
async def get_users():
    return [{'id': 1, 'name': 'Alice'}]
```

### Database

SQL databases (PostgreSQL, MySQL) for relational data.
NoSQL (MongoDB, Redis) for document/cache.

ORM examples:
- SQLAlchemy (Python)
- Prisma (Node.js)
- GORM (Go)

## Infrastructure

### Containerization

Docker 灏佽鎳夌敤鍜屼緷璩?

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### CI/CD

GitHub Actions 绡勪緥:
```yaml
name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: npm install
      - run: npm test
```

## 闆茬鏈嶅嫏

- AWS: EC2, S3, Lambda, RDS
- GCP: Compute Engine, Cloud Storage, Cloud Functions
- Azure: Virtual Machines, Blob Storage

甯歌鏋舵:
- Load Balancer 鈫?App Servers 鈫?Database
- CDN for static assets
- Cache layer (Redis)""",
]


# ---------------------------------------------------------------------------
# 鏁稿鎺ㄧ悊妯ｆ湰
# ---------------------------------------------------------------------------
MATH_TEMPLATES = [
    """Problem: 璀夋槑 sqrt(2) 鏄劇鐞嗘暩.

Proof by contradiction.

鍋囪ō sqrt(2) 鏄湁鐞嗘暩, 鍓?sqrt(2) = p/q, 鍏朵腑 p, q 鐐轰簰璩暣鏁? q 鈮?0.

鍏╅倞骞虫柟: 2 = p虏/q虏
鎵€浠?p虏 = 2q虏.

鍥犳 p虏 鏄伓鏁? 鎺ㄥ緱 p 鏄伓鏁?(鍥犵偤濂囨暩骞虫柟浠嶆槸濂囨暩).
瑷?p = 2k, 浠ｅ叆: (2k)虏 = 2q虏 鈫?4k虏 = 2q虏 鈫?q虏 = 2k虏.

鎵€浠?q虏 涔熸槸鍋舵暩, 鎺ㄥ緱 q 鏄伓鏁?

浣嗛€欒垏 p, q 浜掕唱鐭涚浘 (鍏╄€呴兘鏄伓鏁?. 鏁呭亣瑷尟瑾? sqrt(2) 鏄劇鐞嗘暩. QED.""",

    """Problem: 姹傚嚱鏁?f(x) = x鲁 - 6x虏 + 9x + 1 鐨勬サ鍊?

Step 1: 姹傚皫鏁?
f'(x) = 3x虏 - 12x + 9

Step 2: 浠?f'(x) = 0.
3x虏 - 12x + 9 = 0
x虏 - 4x + 3 = 0
(x - 1)(x - 3) = 0

Critical points: x = 1, x = 3.

Step 3: 浜岄殠灏庢暩鍒ゅ垾.
f''(x) = 6x - 12

At x = 1: f''(1) = 6 - 12 = -6 < 0 鈫?local maximum
  f(1) = 1 - 6 + 9 + 1 = 5

At x = 3: f''(3) = 18 - 12 = 6 > 0 鈫?local minimum
  f(3) = 27 - 54 + 27 + 1 = 1

Answer: local max at (1, 5), local min at (3, 1).""",

    """Problem: 姗熺巼椤?- 钂欐彁闇嶇埦鍟忛.

鏈変笁鎵囬杸, 涓€鎵囧緦闈㈡槸杌? 鍏╂墖寰岄潰鏄北缇? 浣犻伕涓€鎵囬杸 (闁€ 1). 涓绘寔浜?(鐭ラ亾杌婂湪鍝? 鎵撻枊鍙︿竴鎵囨湁灞辩緤鐨勯杸 (闁€ 3). 鍟? 浣犳噳瑭叉彌鍒伴杸 2 鍡?

Solution: 鎳夎┎鎻?

鍒嗘瀽鎵€鏈?3 绋瓑鍙兘鎯呮硜 (杌婂湪闁€ 1/2/3):

Case 1: 杌婂湪闁€ 1 (浣犻伕鐨?
  涓绘寔浜洪枊闁€ 3 (灞辩緤). 鎻?鈫?寰楀埌闁€ 2 (灞辩緤). 杓?

Case 2: 杌婂湪闁€ 2
  涓绘寔浜哄繀闋堥枊闁€ 3 (闁€ 1 浣犻伕浜? 闁€ 2 鏄粖). 鎻?鈫?寰楀埌闁€ 2 (杌?. 璐?

Case 3: 杌婂湪闁€ 3
  涓绘寔浜哄繀闋堥枊闁€ 2. 鎻?鈫?寰楀埌闁€ 3 (杌?. 璐?

P(鎻涗簡璐? = 2/3, P(涓嶆彌璐? = 1/3.

鎵€浠ユ彌鐨勭瓥鐣ュ嫕鐜?2/3, 涓嶆彌鍙湁 1/3.""",

    """Problem: 瑷堢畻 sum_{k=1}^{n} k虏 鐨勫叕寮?

宸茬煡: sum k = n(n+1)/2

Method: 浣跨敤 (k+1)鲁 - k鲁 = 3k虏 + 3k + 1 姹傚拰.

Sum both sides from k=1 to n:
  (n+1)鲁 - 1鲁 = 3路sum(k虏) + 3路sum(k) + sum(1)

(n+1)鲁 - 1 = 3路S + 3路n(n+1)/2 + n
where S = sum(k虏).

Solve for S:
3S = (n+1)鲁 - 1 - 3n(n+1)/2 - n
   = n鲁 + 3n虏 + 3n - 3n(n+1)/2 - n
   = n鲁 + 3n虏 + 2n - 3n(n+1)/2
   = [2n鲁 + 6n虏 + 4n - 3n虏 - 3n] / 2
   = [2n鲁 + 3n虏 + n] / 2
   = n(2n虏 + 3n + 1) / 2
   = n(n+1)(2n+1) / 2

S = n(n+1)(2n+1) / 6.

Verify: n=2: 1+4=5. Formula: 2路3路5/6 = 5. 鉁?"",
]


# ---------------------------------------------------------------------------
# 鐢熸垚涓荤▼寮?
# ---------------------------------------------------------------------------
CATEGORIES = {
    "code": CODE_TEMPLATES,
    "dialog": DIALOG_TEMPLATES,
    "mixed": MIXED_TEMPLATES,
    "long": LONG_TEMPLATES,
    "math": MATH_TEMPLATES,
}


def gen_corpus(output_dir: str, per_category: int = 20):
    """鐢熸垚 corpus 璩囨枡澶?"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    total = 0
    for cat, templates in CATEGORIES.items():
        for i in range(per_category):
            # 寰炴ā鏉夸腑閬镐竴鍊? 鍔犱笂闅ㄦ璁婂寲 (鍓嶇洞瑷婚噵)
            tpl = random.choice(templates)
            # 闅ㄦ鍔犱竴浜涜畩鍖? 搴忚櫉/鏅傞枔鎴?
            variation = f"# sample-{cat}-{i:03d}\n# generated: 2026-08-12\n\n"
            content = variation + tpl

            # 鍓獢鍚?
            if cat == "code":
                # 鏍规摎鍏у姹哄畾鍓獢鍚?
                if "def " in tpl and "python" in tpl.lower():
                    ext = ".py"
                elif "function" in tpl and "javascript" in tpl.lower():
                    ext = ".js"
                elif "struct" in tpl and "swift" in tpl.lower():
                    ext = ".swift"
                elif "interface" in tpl and "typescript" in tpl.lower():
                    ext = ".ts"
                else:
                    ext = ".py"
            else:
                ext = ".md" if cat in ("mixed", "long") else ".txt"

            filename = f"{cat}_{i:03d}{ext}"
            filepath = out / filename
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            total += 1

    print(f"鐢熸垚瀹屾垚: {total} 鍊嬫獢妗?in {output_dir}/")
    print(f"  code:   {per_category} 鍊?(.py/.js/.swift/.ts)")
    print(f"  dialog: {per_category} 鍊?(.txt)")
    print(f"  mixed:  {per_category} 鍊?(.md)")
    print(f"  long:   {per_category} 鍊?(.md)")
    print(f"  math:   {per_category} 鍊?(.txt)")
    print()
    print("涓嬩竴姝?")
    print(f"  py collect_batch.py \\")
    print(f"    --gemma4-url http://192.168.101.X:8080 \\")
    print(f"    --qwen36-url http://192.168.101.Y:8080 \\")
    print(f"    --input {output_dir} \\")
    print(f"    --output train.pt")


def main():
    parser = argparse.ArgumentParser(description="鐢熸垚 sample corpus")
    parser.add_argument("--output", default="corpus",
                        help="杓稿嚭璩囨枡澶?(default: corpus)")
    parser.add_argument("--per-category", type=int, default=20,
                        help="姣忛鐢熸垚澶氬皯鍊?(default: 20, 鍏?5 椤?= 100 鍊?")
    args = parser.parse_args()

    gen_corpus(args.output, args.per_category)


if __name__ == "__main__":
    main()
