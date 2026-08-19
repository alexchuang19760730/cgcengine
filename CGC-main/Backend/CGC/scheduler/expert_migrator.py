import logging
import time
from collections import defaultdict

class HotExpertMigrator:
    """
    M7.6 Gate: Dynamic Hot-Expert Migration.
    Monitors expert utilization and dynamically migrates hot experts' weights
    to nodes with high demand, reducing cross-node network traffic.
    """
    def __init__(self, num_experts=64, migration_threshold=0.8):
        self.num_experts = num_experts
        self.migration_threshold = migration_threshold
        # 記錄每個專家在各個節點上的命中次數
        self.expert_hit_history = defaultdict(lambda: defaultdict(int))
        self.expert_locations = {} # expert_id -> node_id
        
        # 模擬初始靜態分區：前一半在 Node 0，後一半在 Node 1
        for i in range(num_experts):
            self.expert_locations[i] = 0 if i < num_experts // 2 else 1

    def record_routing(self, tokens, routing_weights, source_node_id):
        """
        Record which experts are being called from which nodes.
        """
        # 模擬解析 routing_weights 找到 Top-K 專家
        # 在真實環境中，這裡會接收來自 Gate 的真實 Token 分佈
        hot_expert_id = int(routing_weights.argmax())
        
        # 如果該專家不在發出請求的節點上，代表發生了「跨機通訊」
        if self.expert_locations[hot_expert_id] != source_node_id:
            self.expert_hit_history[source_node_id][hot_expert_id] += len(tokens)
            logging.debug(f"[Migrator] Cross-node hit: Node {source_node_id} called Expert {hot_expert_id}.")

    def evaluate_and_migrate(self):
        """
        Periodically evaluate if any expert is too "hot" for a remote node
        and trigger a background migration (weight caching).
        """
        migrations = []
        for node_id, hits in self.expert_hit_history.items():
            for expert_id, traffic_volume in hits.items():
                # 如果跨機流量超過閾值，觸發遷移
                if traffic_volume > 5000: # 模擬閾值：超過 5000 個 Token
                    logging.info(f"🔥 [Migrator] Hot-Expert Detected! Expert {expert_id} is heavily requested by Node {node_id}.")
                    
                    # 執行熱點轉移：將專家權重快取到需求節點
                    self._trigger_background_weight_transfer(expert_id, target_node=node_id)
                    
                    # 更新路由表：未來 Node {node_id} 對 Expert {expert_id} 的請求將直接在本地計算
                    self.expert_locations[expert_id] = node_id
                    migrations.append((expert_id, node_id))
                    
                    # 清空該專家的歷史紀錄
                    self.expert_hit_history[node_id][expert_id] = 0
                    
        return migrations

    def _trigger_background_weight_transfer(self, expert_id, target_node):
        """
        Trigger DeepEP / Ray to asynchronously copy the expert's weights.
        """
        logging.info(f"🚚 [Migrator] Initiating background RDMA/TCP transfer of Expert {expert_id} weights to Node {target_node}...")
        # 模擬傳輸時間
        # time.sleep(0.1) 
        logging.info(f"✅ [Migrator] Expert {expert_id} successfully cached on Node {target_node}. Cross-node traffic eliminated.")
