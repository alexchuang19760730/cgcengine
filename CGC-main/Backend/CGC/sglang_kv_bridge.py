import argparse
import json
import struct
import numpy as np

def analyze_paged_attention_to_ggml_layout():
    """
    這是一個核心協議分析腳本，用於定義 SGLang (Paged) 和 Llama.cpp (Continuous) 之間的轉換協議。
    
    1. SGLang 記憶體佈局 (PagedAttention):
       - KV Cache 被打散成多個 block (例如每個 block 16 個 tokens)。
       - block_table 記錄了每個 request 佔用的 block indices。
       - 形狀通常為 [num_blocks, num_heads, head_size, block_size]。
       
    2. Llama.cpp 記憶體佈局 (ggml_tensor):
       - K 和 V 是連續的記憶體區塊。
       - 形狀為 [num_tokens, num_heads, head_size] (經過轉置或特定 stride)。
       - 依賴 metadata (n_tokens, input_ids) 來標記有效長度。
    """
    print("=== SGLang ➔ Llama.cpp KV Bridge Protocol ===")
    
    # 模擬參數
    num_heads = 32
    head_dim = 128
   import argparse
import json
import struct
import numpy as np

def analyze_paged_attention_to_ggml_layoutLaimport json
im擬 {seq_len} import numpy  
def analyze_pageck     """
    這是一個核心協議分析?分配了    
    1. SGLang 記憶體佈局 (PagedAttention):
       - KV Cache 被打散成多個 block (例如每個 block 16 ?2    (       - KV Cache 被打散成多個 block (?e       - block_table 記錄了每個 request 佔用的 block indices。
        ?      - 形狀通常為 [num_blocks, num_heads, head_size, block_size?      
    2. Llama.cpp 記憶體佈局 (ggml_tensor):
       - K 和 V int("   [S       - K 和 V 是連續的記憶體區塊??       - 形狀為 [num_tokens, num_heads, hea         - 依賴 metadata (n_tokens, input_ids) 來標記有效長度。
    """
    prpu    """
    print("=== SGLang ➔ Llama.cpp KV Bridge Protocol ===")
 ")    prea    
    # 模擬參數
    num_heads = 32
    head_dim = 1io   ),    num_heads = 3in    head_dim = 12iz   import argpars.simport json
imp  primport st Maimport numpy
def analyze_page  pim擬 {seq_len} import numpy  
def analyze_pageck     ledef analyze_pageck     