#!/usr/bin/env python3
"""生成多樣化 sample corpus, 用於 MoT-h 訓練對採集.

生成 5 類文本, 每類 N 條, 寫入 corpus/ 資料夾.
覆蓋: 代碼 / 對話 / 中英混 / 長文 / 數學推理.

用法:
  py gen_sample_corpus.py --output corpus/ --per-category 20

輸出:
  corpus/
    ├── code_001.py
    ├── code_002.py
    ...
    ├── dialog_001.txt
    ...
    ├── mixed_001.txt
    ...
    ├── long_001.md
    ...
    └── math_001.txt
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------------------------
# 代碼樣本 (Python/Swift/JS)
# ---------------------------------------------------------------------------
CODE_TEMPLATES = [
    """# Fibonacci 數列 - 迭代版
def fibonacci(n: int) -> list[int]:
    \"\"\"返回前 n 個 Fibonacci 數.\"\"\"
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

    """# 二分搜索
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

    """// JavaScript: 簡單的 debounce 函數
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
log("world");  // 只會印 "world"
""",

    """// Swift: 簡單的 Stack 結構
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

    """# Python: 簡單的 LRU Cache
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

    """# Python: 快速排序
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

    """# Python: 簡單 HTTP server
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

    """// TypeScript: 簡單的 Observable
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
# 對話樣本
# ---------------------------------------------------------------------------
DIALOG_TEMPLATES = [
    """User: 你好, 請問 Python 的 list 和 tuple 有什麼區別?
Assistant: 主要區別有兩點:
1. list 是可變的, tuple 是不可變的
2. tuple 通常用於異構數據 (如座標), list 用於同構數據

例如:
  coords = (3, 4)       # tuple, 不會變
  points = [1, 2, 3]    # list, 可以 append/pop

User: 那性能上有差別嗎?
Assistant: tuple 因為不可變, 創建速度稍快, 內存佔用略小. 但在大部分場景下差異不顯著.""",

    """User: 解釋一下什麼是閉包 (closure).
Assistant: 閉包是函數 + 其詞法環境的組合. 內層函數可以訪問外層函數的變量, 即使外層函數已經返回.

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

這裡 `inc` 閉包了 `count` 變量.

User: 為什麼要 nonlocal?
Assistant: 因為 Python 默認把賦值當作創建局部變量. nonlocal 告訴 Python 這個變量在外層作用域.""",

    """User: What is the time complexity of binary search?
Assistant: O(log n) where n is the array length. Each comparison halves the search space.

For example, searching a sorted array of 1 million elements takes at most ~20 comparisons (log2(10^6) ≈ 20).

User: What about space complexity?
Assistant: O(1) for the iterative version, O(log n) for the recursive version (call stack).""",

    """User: 解釋一下 RESTful API.
Assistant: REST 是一種 API 設計風格, 核心原則:
- 用 HTTP method 表達操作: GET (查), POST (建), PUT (改), DELETE (刪)
- 用 URL 表達資源: /users/123
- 無狀態: 每個請求自包含

例如:
  GET    /users        # 列出所有用戶
  GET    /users/123    # 取得單個用戶
  POST   /users        # 創建用戶
  PUT    /users/123    # 更新用戶
  DELETE /users/123    # 刪除用戶

User: 那 PATCH 呢?
Assistant: PATCH 用於部分更新, 只發送變化的字段. PUT 是全量替換.""",
]


# ---------------------------------------------------------------------------
# 中英混合長文
# ---------------------------------------------------------------------------
MIXED_TEMPLATES = [
    """# 機器學習基礎

Machine learning is a subset of artificial intelligence that enables systems to learn from data.

## 監督式學習 (Supervised Learning)

監督式學習使用標記數據訓練模型. 常見任務包括:
- Classification: 分類問題, 如圖片辨識
- Regression: 回歸問題, 如房價預測

例如, 給定一批貓狗圖片和標籤, 訓練一個分類器:
```python
from sklearn.svm import SVC
clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

## 非監督式學習 (Unsupervised Learning)

非監督式學習處理無標籤數據, 常見方法:
- Clustering: 聚類, 如 K-means
- Dimensionality reduction: 降維, 如 PCA

## 深度學習 (Deep Learning)

深度學習使用多層神經網絡. 主要類型:
1. CNN (Convolutional Neural Network) - 圖像
2. RNN (Recurrent Neural Network) - 序列
3. Transformer - 語言模型 (如 GPT, BERT)

Transformer 的核心是 self-attention 機制, 讓模型能關注序列中任意位置的 token.""",

    """# 計算機網路基礎

Computer networks connect devices to share resources and communicate.

## OSI 七層模型

The OSI model has 7 layers:
1. Physical (物理層) - 位元傳輸
2. Data Link (資料鏈結層) - 幀傳輸, MAC 定址
3. Network (網路層) - IP 路由
4. Transport (傳輸層) - TCP/UDP, 端口
5. Session (會議層) - 會話管理
6. Presentation (表達層) - 加密, 壓縮
7. Application (應用層) - HTTP, FTP, SMTP

## TCP vs UDP

TCP (Transmission Control Protocol):
- 可靠傳輸, 三次握手建立連接
- 有序, 無丟包
- 應用: HTTP, SMTP, FTP

UDP (User Datagram Protocol):
- 不可靠, 無連接
- 速度快, 可丟包
- 應用: DNS, 視頻流, 遊戲

## HTTP 協議

HTTP 是無狀態的 request-response 協議. 常見狀態碼:
- 200 OK
- 301 Moved Permanently
- 404 Not Found
- 500 Internal Server Error""",

    """# 資料結構與演算法

Data structures organize data for efficient access. Algorithms solve problems step by step.

## 基本資料結構

### Array (陣列)
- 連續內存, O(1) 隨機存取
- 插入/刪除 O(n)

### Linked List (鏈結串列)
- 非連續, 通過指針連接
- 插入/刪除 O(1) (已知節點)
- 存取 O(n)

### Hash Table (雜湊表)
- Key-value 映射
- 平均 O(1) 查找/插入/刪除
- Python dict, JavaScript Map

### Tree (樹)
- 階層結構
- 二元搜索樹: 左 < 根 < 右
- 平衡樹: AVL, Red-Black

## 排序演算法

| Algorithm | Time (avg) | Time (worst) | Space | Stable |
|-----------|------------|--------------|-------|--------|
| Quicksort | O(n log n) | O(n²)        | O(log n) | No  |
| Mergesort | O(n log n) | O(n log n)   | O(n)  | Yes    |
| Heapsort  | O(n log n) | O(n log n)   | O(1)  | No     |
| Bubble    | O(n²)      | O(n²)        | O(1)  | Yes    |

## 搜尋演算法

- Linear search: O(n)
- Binary search: O(log n) (需排序)
- Hash lookup: O(1) 平均""",
]


# ---------------------------------------------------------------------------
# 長文樣本
# ---------------------------------------------------------------------------
LONG_TEMPLATES = [
    """# 軟體工程原則

Software engineering is the systematic application of engineering principles to software.

## 1. 單一職責原則 (Single Responsibility Principle)

一個類別或模組應該只有一個職責. 這意味著它只有一個變化的原因.

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

## 2. 開放封閉原則 (Open/Closed Principle)

軟體實體應該對擴展開放, 對修改封閉. 通過繼承或組合添加新功能, 而不是修改現有代碼.

## 3. 里氏替換原則 (Liskov Substitution Principle)

子類別必須能替換父類別而不破壞程式行為. 子類別不應該加更嚴格的前置條件, 或更弱的後置條件.

## 4. 介面隔離原則 (Interface Segregation Principle)

客戶端不應該被迫依賴它不使用的方法. 多個專用介面優於一個通用介面.

## 5. 依賴反轉原則 (Dependency Inversion Principle)

高層模組不應該依賴低層模組. 兩者都應該依賴抽象. 抽象不應該依賴細節, 細節應該依賴抽象.

## 設計模式

### 創建型模式
- Singleton: 確保一個類只有一個實例
- Factory Method: 定義創建物件的介面, 讓子類決定實例化哪個類
- Abstract Factory: 創建一系列相關物件
- Builder: 分步驟構建複雜物件

### 結構型模式
- Adapter: 轉換介面
- Decorator: 動態添加職責
- Facade: 提供統一介面
- Proxy: 控制存取

### 行為型模式
- Observer: 發布-訂閱
- Strategy: 封裝可互換的演算法
- Command: 將請求封裝為物件
- Iterator: 順序存取聚合元素

## 測試驅動開發 (TDD)

TDD 流程: Red → Green → Refactor
1. Red: 寫一個失敗的測試
2. Green: 寫最少代碼讓測試通過
3. Refactor: 改進代碼結構

優點:
- 更清晰的介面設計
- 即時反饋
- 重構信心
- 文檔化預期行為""",

    """# 現代 Web 開發

Modern web development spans frontend, backend, and infrastructure.

## Frontend

### HTML/CSS/JavaScript 基礎

HTML 結構化內容, CSS 控制樣式, JavaScript 添加互動.

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

React 是聲明式 UI 庫, 使用組件和虛擬 DOM.

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
- useState: 狀態管理
- useEffect: 副作用
- useContext: 全域狀態
- useMemo/useCallback: 性能優化

### Vue

Vue 是漸進式框架, 模板語法更接近 HTML.

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

Docker 封裝應用和依賴.

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### CI/CD

GitHub Actions 範例:
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

## 雲端服務

- AWS: EC2, S3, Lambda, RDS
- GCP: Compute Engine, Cloud Storage, Cloud Functions
- Azure: Virtual Machines, Blob Storage

常見架構:
- Load Balancer → App Servers → Database
- CDN for static assets
- Cache layer (Redis)""",
]


# ---------------------------------------------------------------------------
# 數學推理樣本
# ---------------------------------------------------------------------------
MATH_TEMPLATES = [
    """Problem: 證明 sqrt(2) 是無理數.

Proof by contradiction.

假設 sqrt(2) 是有理數, 則 sqrt(2) = p/q, 其中 p, q 為互質整數, q ≠ 0.

兩邊平方: 2 = p²/q²
所以 p² = 2q².

因此 p² 是偶數, 推得 p 是偶數 (因為奇數平方仍是奇數).
設 p = 2k, 代入: (2k)² = 2q² → 4k² = 2q² → q² = 2k².

所以 q² 也是偶數, 推得 q 是偶數.

但這與 p, q 互質矛盾 (兩者都是偶數). 故假設錯誤, sqrt(2) 是無理數. QED.""",

    """Problem: 求函數 f(x) = x³ - 6x² + 9x + 1 的極值.

Step 1: 求導數.
f'(x) = 3x² - 12x + 9

Step 2: 令 f'(x) = 0.
3x² - 12x + 9 = 0
x² - 4x + 3 = 0
(x - 1)(x - 3) = 0

Critical points: x = 1, x = 3.

Step 3: 二階導數判別.
f''(x) = 6x - 12

At x = 1: f''(1) = 6 - 12 = -6 < 0 → local maximum
  f(1) = 1 - 6 + 9 + 1 = 5

At x = 3: f''(3) = 18 - 12 = 6 > 0 → local minimum
  f(3) = 27 - 54 + 27 + 1 = 1

Answer: local max at (1, 5), local min at (3, 1).""",

    """Problem: 機率題 - 蒙提霍爾問題.

有三扇門, 一扇後面是車, 兩扇後面是山羊. 你選一扇門 (門 1). 主持人 (知道車在哪) 打開另一扇有山羊的門 (門 3). 問: 你應該換到門 2 嗎?

Solution: 應該換.

分析所有 3 種等可能情況 (車在門 1/2/3):

Case 1: 車在門 1 (你選的)
  主持人開門 3 (山羊). 換 → 得到門 2 (山羊). 輸.

Case 2: 車在門 2
  主持人必須開門 3 (門 1 你選了, 門 2 是車). 換 → 得到門 2 (車). 贏.

Case 3: 車在門 3
  主持人必須開門 2. 換 → 得到門 3 (車). 贏.

P(換了贏) = 2/3, P(不換贏) = 1/3.

所以換的策略勝率 2/3, 不換只有 1/3.""",

    """Problem: 計算 sum_{k=1}^{n} k² 的公式.

已知: sum k = n(n+1)/2

Method: 使用 (k+1)³ - k³ = 3k² + 3k + 1 求和.

Sum both sides from k=1 to n:
  (n+1)³ - 1³ = 3·sum(k²) + 3·sum(k) + sum(1)

(n+1)³ - 1 = 3·S + 3·n(n+1)/2 + n
where S = sum(k²).

Solve for S:
3S = (n+1)³ - 1 - 3n(n+1)/2 - n
   = n³ + 3n² + 3n - 3n(n+1)/2 - n
   = n³ + 3n² + 2n - 3n(n+1)/2
   = [2n³ + 6n² + 4n - 3n² - 3n] / 2
   = [2n³ + 3n² + n] / 2
   = n(2n² + 3n + 1) / 2
   = n(n+1)(2n+1) / 2

S = n(n+1)(2n+1) / 6.

Verify: n=2: 1+4=5. Formula: 2·3·5/6 = 5. ✓""",
]


# ---------------------------------------------------------------------------
# 生成主程式
# ---------------------------------------------------------------------------
CATEGORIES = {
    "code": CODE_TEMPLATES,
    "dialog": DIALOG_TEMPLATES,
    "mixed": MIXED_TEMPLATES,
    "long": LONG_TEMPLATES,
    "math": MATH_TEMPLATES,
}


def gen_corpus(output_dir: str, per_category: int = 20):
    """生成 corpus 資料夾."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    total = 0
    for cat, templates in CATEGORIES.items():
        for i in range(per_category):
            # 從模板中選一個, 加上隨機變化 (前綴註釋)
            tpl = random.choice(templates)
            # 隨機加一些變化: 序號/時間戳
            variation = f"# sample-{cat}-{i:03d}\n# generated: 2026-08-12\n\n"
            content = variation + tpl

            # 副檔名
            if cat == "code":
                # 根據內容決定副檔名
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

    print(f"生成完成: {total} 個檔案 in {output_dir}/")
    print(f"  code:   {per_category} 個 (.py/.js/.swift/.ts)")
    print(f"  dialog: {per_category} 個 (.txt)")
    print(f"  mixed:  {per_category} 個 (.md)")
    print(f"  long:   {per_category} 個 (.md)")
    print(f"  math:   {per_category} 個 (.txt)")
    print()
    print("下一步:")
    print(f"  py collect_batch.py \\")
    print(f"    --gemma4-url http://192.168.101.X:8080 \\")
    print(f"    --qwen36-url http://192.168.101.Y:8080 \\")
    print(f"    --input {output_dir} \\")
    print(f"    --output train.pt")


def main():
    parser = argparse.ArgumentParser(description="生成 sample corpus")
    parser.add_argument("--output", default="corpus",
                        help="輸出資料夾 (default: corpus)")
    parser.add_argument("--per-category", type=int, default=20,
                        help="每類生成多少個 (default: 20, 共 5 類 = 100 個)")
    args = parser.parse_args()

    gen_corpus(args.output, args.per_category)


if __name__ == "__main__":
    main()
