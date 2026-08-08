#!/usr/bin/env python3
"""生成 MTP 训练用的 corpus (JSONL 格式).

每个 entry 是 {"text": "..."} 格式, 内容为代码/技术文本.
用于 Gemma4-26B MTP head decode-mode 训练.
"""
import json
import random

# 代码片段
CODE_SNIPPETS = [
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
    "class LinkedList:\n    def __init__(self):\n        self.head = None\n    def append(self, val):\n        if not self.head:\n            self.head = Node(val)\n            return\n        cur = self.head\n        while cur.next:\n            cur = cur.next\n        cur.next = Node(val)",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)",
    "async function fetchData(url) {\n  try {\n    const response = await fetch(url);\n    if (!response.ok) throw new Error('Network error');\n    const data = await response.json();\n    return data;\n  } catch (error) {\n    console.error('Fetch failed:', error);\n    return null;\n  }\n}",
    "import numpy as np\n\ndef normalize(matrix):\n    mean = np.mean(matrix, axis=0)\n    std = np.std(matrix, axis=0)\n    return (matrix - mean) / (std + 1e-8)",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b",
    "class Queue:\n    def __init__(self):\n        self.items = []\n    def enqueue(self, item):\n        self.items.append(item)\n    def dequeue(self):\n        if not self.is_empty():\n            return self.items.pop(0)\n    def is_empty(self):\n        return len(self.items) == 0",
    "def dfs(graph, start, visited=None):\n    if visited is None:\n        visited = set()\n    visited.add(start)\n    for neighbor in graph[start]:\n        if neighbor not in visited:\n            dfs(graph, neighbor, visited)\n    return visited",
    "function debounce(fn, delay) {\n  let timer;\n  return function(...args) {\n    clearTimeout(timer);\n    timer = setTimeout(() => fn.apply(this, args), delay);\n  };\n}",
    "def train_model(model, data, epochs, lr):\n    optimizer = torch.optim.Adam(model.parameters(), lr=lr)\n    for epoch in range(epochs):\n        for batch in data:\n            loss = model(batch)\n            loss.backward()\n            optimizer.step()\n            optimizer.zero_grad()",
    "import threading\n\nclass ThreadSafeCounter:\n    def __init__(self):\n        self._count = 0\n        self._lock = threading.Lock()\n    def increment(self):\n        with self._lock:\n            self._count += 1\n    def get(self):\n        with self._lock:\n            return self._count",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
    "export default function Button({ label, onClick }) {\n  return (\n    <button onClick={onClick} className=\"btn\">\n      {label}\n    </button>\n  );\n}",
    "def validate_email(email):\n    import re\n    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n    return bool(re.match(pattern, email))",
    "class TrieNode:\n    def __init__(self):\n        self.children = {}\n        self.is_end = False\n\nclass Trie:\n    def __init__(self):\n        self.root = TrieNode()\n    def insert(self, word):\n        node = self.root\n        for char in word:\n            if char not in node.children:\n                node.children[char] = TrieNode()\n            node = node.children[char]\n        node.is_end = True",
]

# 技术解释
EXPLANATIONS = [
    "Binary search is an efficient algorithm for finding a target value in a sorted array. It works by repeatedly dividing the search interval in half. The time complexity is O(log n) because each comparison eliminates half of the remaining elements.",
    "QuickSort is a divide-and-conquer algorithm. It picks a pivot element and partitions the array around the pivot. The average time complexity is O(n log n), but the worst case is O(n^2) when the pivot is always the smallest or largest element.",
    "Merge sort is a stable sorting algorithm that uses divide and conquer. It divides the array into two halves, recursively sorts them, and then merges the sorted halves. The time complexity is always O(n log n).",
    "A hash table uses a hash function to map keys to indices in an array. When collisions occur, they can be resolved using chaining or open addressing. The average time complexity for insert, delete, and lookup is O(1).",
    "Dynamic programming solves problems by breaking them into overlapping subproblems and storing results to avoid recomputation. It is useful for optimization problems like the knapsack problem, longest common subsequence, and edit distance.",
    "A binary tree is a tree data structure where each node has at most two children. A binary search tree maintains the property that left child values are less than the parent, and right child values are greater.",
    "Depth-first search (DFS) traverses a graph by going as deep as possible before backtracking. It uses a stack or recursion. Breadth-first search (BFS) explores all neighbors at the current depth before moving deeper, using a queue.",
    "Object-oriented programming organizes code around objects that combine data and behavior. The four pillars are encapsulation, inheritance, polymorphism, and abstraction.",
    "Recursion is when a function calls itself. Every recursive function needs a base case to terminate. The call stack keeps track of each recursive call. Too many recursive calls can lead to a stack overflow.",
    "Big O notation describes the upper bound of an algorithm's time or space complexity. Common complexities from best to worst: O(1), O(log n), O(n), O(n log n), O(n^2), O(2^n), O(n!).",
]

# 错误修复
ERROR_FIXES = [
    "IndexError: list index out of range. This occurs when you try to access an index that does not exist. Fix: check len(list) before accessing or use try-except.",
    "KeyError: 'key'. This happens when you try to access a dictionary key that does not exist. Fix: use dict.get(key) or check if key in dict first.",
    "TypeError: 'NoneType' object is not subscriptable. This means you are trying to index into None. Fix: add a None check before accessing the variable.",
    "AttributeError: 'str' object has no attribute 'append'. Strings are immutable in Python. Fix: use string concatenation or convert to list first.",
    "RecursionError: maximum recursion depth exceeded. This means your recursive function lacks a proper base case. Fix: ensure the base case is reachable.",
    "ValueError: invalid literal for int(). This occurs when trying to convert a non-numeric string to int. Fix: validate input or use try-except.",
    "ZeroDivisionError: division by zero. Fix: check if the denominator is zero before dividing.",
    "ImportError: No module named 'module'. Fix: install the package with pip install or check the module path.",
]

# 算法描述
ALGORITHMS = [
    "The A* search algorithm finds the shortest path between nodes. It uses a heuristic function to estimate the cost to reach the goal, combining the actual cost from the start with the estimated cost to the goal.",
    "Dijkstra's algorithm finds the shortest path from a source node to all other nodes in a weighted graph. It uses a priority queue to always expand the node with the smallest known distance.",
    "The Knapsack problem can be solved with dynamic programming. Create a 2D table where dp[i][w] represents the maximum value using the first i items with weight limit w.",
    "The traveling salesman problem (TSP) asks for the shortest route visiting all cities exactly once. It is NP-hard, but heuristic approaches like nearest neighbor and 2-opt can find good solutions.",
    "Breadth-first search can find the shortest path in an unweighted graph. It explores nodes level by level, guaranteeing the first time we reach a node is via the shortest path.",
    "The Bellman-Ford algorithm computes shortest paths from a source node, and can handle negative edge weights. It relaxes all edges V-1 times.",
]

def generate_corpus(count=500):
    """Generate diverse corpus entries."""
    entries = []
    for i in range(count):
        category = random.choice(["code", "explain", "fix", "algo"])
        if category == "code":
            text = random.choice(CODE_SNIPPETS)
            # Add some variation
            if random.random() < 0.3:
                text = "# " + text.split('\n')[0] + "\n" + text
        elif category == "explain":
            text = random.choice(EXPLANATIONS)
        elif category == "fix":
            text = random.choice(ERROR_FIXES)
        else:
            text = random.choice(ALGORITHMS)
        
        # Add some prefix variation
        prefixes = ["", "Here is ", "Let me explain. ", "Sure! ", ""]
        text = random.choice(prefixes) + text
        
        entries.append({"text": text})
    return entries


if __name__ == "__main__":
    import sys
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    output = sys.argv[2] if len(sys.argv) > 2 else "/data/mtp_corpus.jsonl"
    
    entries = generate_corpus(count)
    with open(output, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"Generated {len(entries)} corpus entries -> {output}")
