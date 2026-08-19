#!/usr/bin/env python3
"""
CGC Engine Scheduling Layer
Responsible for: expert loading strategy, timing decisions
"""

from typing import List, Dict, Set
import time
from collections import defaultdict

class ExpertScheduler:
    def __init__(self, max_cached_experts: int = 8, prefetch_enabled: bool = True, prefetch_window: int = 32):
        self.max_cached_experts = max_cached_experts
        self.prefetch_enabled = prefetch_enabled
        self.prefetch_window = prefetch_window
        self.access_history = defaultdict(list)
        self.last_access_time = {}
        
    def schedule_experts(self, predicted_experts: List[int], current_cached: Set[int]) -> Dict[str, List[int]]:
        predicted_set = set(predicted_experts)
        to_load = predicted_set - current_cached
        to_keep = predicted_set & current_cached
        candidates_to_unload = current_cached - predicted_set
        required_unload = len(current_cached) + len(to_load) - self.max_cached_experts
        to_unload = []
        
        if required_unload > 0:
            sorted_candidates = sorted(candidates_to_unload, key=lambda eid: self.last_access_time.get(eid, 0))
            to_unload = sorted_candidates[:required_unload]
        
        return {'load': list(to_load), 'unload': to_unload, 'keep': list(to_keep)}
    
    def record_access(self, expert_ids: List[int]):
        now = time.time()
        for eid in expert_ids:
            self.last_access_time[eid] = now
            self.access_history[eid].append(now)
    
    def get_hot_experts(self, top_n: int = 4) -> List[int]:
        expert_access_count = {eid: len(times) for eid, times in self.access_history.items()}
        sorted_experts = sorted(expert_access_count.keys(), key=lambda eid: expert_access_count[eid], reverse=True)
        return sorted_experts[:top_n]
    
    def prefetch_experts(self, current_batch: List[int], next_batch_prediction: List[int]) -> List[int]:
        if not self.prefetch_enabled:
            return []
        current_set = set(current_batch)
        next_set = set(next_batch_prediction)
        to_prefetch = next_set - current_set
        return list(to_prefetch)
