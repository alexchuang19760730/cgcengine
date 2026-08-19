# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
测试结果存储系统

提供:
1. 测试结果持久化存储 (SQLite)
2. 性能指标查询接口
3. 数据导出功能 (CSV/JSON)
4. 统计分析功能
"""

import sqlite3
import json
import csv
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import os

from cgc_engine.utils.envs import cgc_output_dir

@dataclass
class TestRecord:
    """测试记录"""
    test_id: str
    module_name: str
    device: str
    backend: str
    success: bool
    timestamp: str
    
    # 时间指标
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    min_time_ms: float = 0.0
    max_time_ms: float = 0.0
    
    # 内存指标
    peak_memory_gb: float = 0.0
    avg_memory_gb: float = 0.0
    
    # IO 指标
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    copy_count: int = 0
    
    # 计算指标
    gflops: float = 0.0
    total_ops: int = 0
    
    # 调度指标
    scheduling_delay_ms: float = 0.0
    overhead_ratio: float = 0.0
    
    # 硬件信息
    platform: str = ""
    device_type: str = ""
    device_count: int = 0
    total_memory_gb_sys: float = 0.0
    unified_memory: bool = False
    
    # 额外元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class TestResultStorage:
    """测试结果存储系统"""
    
    def __init__(self, db_path: str | None = None):
        """初始化存储系统"""
        self.db_path = db_path or os.path.join(cgc_output_dir(), "test_results.db")
        self._init_database()
    
    def _init_database(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建测试记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT UNIQUE,
                module_name TEXT,
                device TEXT,
                backend TEXT,
                success BOOLEAN,
                timestamp TEXT,
                total_time_ms REAL,
                avg_time_ms REAL,
                min_time_ms REAL,
                max_time_ms REAL,
                peak_memory_gb REAL,
                avg_memory_gb REAL,
                h2d_bytes INTEGER,
                d2h_bytes INTEGER,
                copy_count INTEGER,
                gflops REAL,
                total_ops INTEGER,
                scheduling_delay_ms REAL,
                overhead_ratio REAL,
                platform TEXT,
                device_type TEXT,
                device_count INTEGER,
                total_memory_gb_sys REAL,
                unified_memory BOOLEAN,
                metadata TEXT,
                error_message TEXT
            )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_module_name ON test_records(module_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_device ON test_records(device)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON test_records(timestamp)')
        
        conn.commit()
        conn.close()
    
    def save_record(self, record: TestRecord):
        """保存测试记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO test_records (
                    test_id, module_name, device, backend, success, timestamp,
                    total_time_ms, avg_time_ms, min_time_ms, max_time_ms,
                    peak_memory_gb, avg_memory_gb,
                    h2d_bytes, d2h_bytes, copy_count,
                    gflops, total_ops,
                    scheduling_delay_ms, overhead_ratio,
                    platform, device_type, device_count, total_memory_gb_sys, unified_memory,
                    metadata, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.test_id, record.module_name, record.device, record.backend, record.success, record.timestamp,
                record.total_time_ms, record.avg_time_ms, record.min_time_ms, record.max_time_ms,
                record.peak_memory_gb, record.avg_memory_gb,
                record.h2d_bytes, record.d2h_bytes, record.copy_count,
                record.gflops, record.total_ops,
                record.scheduling_delay_ms, record.overhead_ratio,
                record.platform, record.device_type, record.device_count, record.total_memory_gb_sys, record.unified_memory,
                json.dumps(record.metadata), record.error_message
            ))
            
            conn.commit()
            print(f"✅ 测试记录已保存: {record.test_id}")
        except Exception as e:
            print(f"❌ 保存失败: {e}")
        finally:
            conn.close()
    
    def get_record(self, test_id: str) -> Optional[TestRecord]:
        """获取单条记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM test_records WHERE test_id = ?', (test_id,))
        row = cursor.fetchone()
        
        if row:
            return self._row_to_record(row)
        
        conn.close()
        return None
    
    def get_records_by_module(self, module_name: str) -> List[TestRecord]:
        """按模块获取记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM test_records WHERE module_name = ? ORDER BY timestamp DESC', (module_name,))
        rows = cursor.fetchall()
        
        records = [self._row_to_record(row) for row in rows]
        conn.close()
        return records
    
    def get_all_records(self) -> List[TestRecord]:
        """获取所有记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM test_records ORDER BY timestamp DESC')
        rows = cursor.fetchall()
        
        records = [self._row_to_record(row) for row in rows]
        conn.close()
        return records
    
    def delete_record(self, test_id: str):
        """删除记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM test_records WHERE test_id = ?', (test_id,))
        conn.commit()
        conn.close()
    
    def _row_to_record(self, row) -> TestRecord:
        """将数据库行转换为 TestRecord"""
        return TestRecord(
            test_id=row[1],
            module_name=row[2],
            device=row[3],
            backend=row[4],
            success=bool(row[5]),
            timestamp=row[6],
            total_time_ms=row[7],
            avg_time_ms=row[8],
            min_time_ms=row[9],
            max_time_ms=row[10],
            peak_memory_gb=row[11],
            avg_memory_gb=row[12],
            h2d_bytes=row[13],
            d2h_bytes=row[14],
            copy_count=row[15],
            gflops=row[16],
            total_ops=row[17],
            scheduling_delay_ms=row[18],
            overhead_ratio=row[19],
            platform=row[20],
            device_type=row[21],
            device_count=row[22],
            total_memory_gb_sys=row[23],
            unified_memory=bool(row[24]),
            metadata=json.loads(row[25]) if row[25] else {},
            error_message=row[26]
        )
    
    def export_to_csv(self, filepath: str, module_name: Optional[str] = None):
        """导出记录到 CSV"""
        records = self.get_records_by_module(module_name) if module_name else self.get_all_records()
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'test_id', 'module_name', 'device', 'backend', 'success', 'timestamp',
                'total_time_ms', 'avg_time_ms', 'peak_memory_gb',
                'h2d_bytes', 'd2h_bytes', 'gflops'
            ])
            
            for record in records:
                writer.writerow([
                    record.test_id, record.module_name, record.device, record.backend, record.success,
                    record.timestamp, record.total_time_ms, record.avg_time_ms,
                    record.peak_memory_gb, record.h2d_bytes, record.d2h_bytes, record.gflops
                ])
        
        print(f"✅ 数据已导出到: {filepath}")
    
    def export_to_json(self, filepath: str, module_name: Optional[str] = None):
        """导出记录到 JSON"""
        records = self.get_records_by_module(module_name) if module_name else self.get_all_records()
        
        data = [{
            'test_id': r.test_id,
            'module_name': r.module_name,
            'device': r.device,
            'backend': r.backend,
            'success': r.success,
            'timestamp': r.timestamp,
            'total_time_ms': r.total_time_ms,
            'avg_time_ms': r.avg_time_ms,
            'peak_memory_gb': r.peak_memory_gb,
            'h2d_bytes': r.h2d_bytes,
            'd2h_bytes': r.d2h_bytes,
            'gflops': r.gflops,
            'metadata': r.metadata
        } for r in records]
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ 数据已导出到: {filepath}")
    
    def get_statistics(self, module_name: Optional[str] = None) -> Dict[str, Any]:
        """获取统计信息"""
        records = self.get_records_by_module(module_name) if module_name else self.get_all_records()
        
        if not records:
            return {'error': 'No records found'}
        
        success_count = sum(1 for r in records if r.success)
        avg_time = sum(r.total_time_ms for r in records) / len(records)
        avg_memory = sum(r.peak_memory_gb for r in records) / len(records)
        avg_gflops = sum(r.gflops for r in records) / len(records)
        
        return {
            'total_records': len(records),
            'success_rate': success_count / len(records) * 100,
            'avg_total_time_ms': avg_time,
            'avg_peak_memory_gb': avg_memory,
            'avg_gflops': avg_gflops,
            'first_record': records[-1].timestamp,
            'latest_record': records[0].timestamp
        }
