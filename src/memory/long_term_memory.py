import lancedb
import sqlite3
import pyarrow as pa
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import logging
import time

logger = logging.getLogger(__name__)

class LongTermMemory:
    def __init__(self, lancedb_path: str, sqlite_path: str):
        # 1. 初始化 LanceDB (向量记忆)
        ldb_path_obj = Path(lancedb_path)
        ldb_path_obj.mkdir(exist_ok=True, parents=True)
        self.vector_db = lancedb.connect(ldb_path_obj)
        
        # 定义 LanceDB schema (只存向量和最核心的关联ID)
        schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), list_size=384)),
            pa.field("event_id", pa.string()),
            pa.field("timestamp", pa.float64()) # 保留时间戳以便快速过滤
        ])
        self.vector_table = self.vector_db.create_table("semantic_memory", schema=schema, exist_ok=True)
        logger.info(f"LanceDB initialized at {lancedb_path}")

        # 2. 初始化 SQLite (结构化知识图谱)
        Path(sqlite_path).parent.mkdir(exist_ok=True, parents=True)
        # check_same_thread=False 允许在不同线程中使用同一个连接对象(需小心锁的问题，这里主要靠应用层控制)
        self.sqlite_conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        self.sqlite_conn.row_factory = sqlite3.Row # 让查询结果可以通过列名访问
        self._init_sqlite_tables()
        logger.info(f"SQLite Knowledge Graph initialized at {sqlite_path}")

        # 3. 加载嵌入模型
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        logger.info("LongTermMemory ready.")

    def _init_sqlite_tables(self):
        """初始化 SQLite 表结构"""
        cursor = self.sqlite_conn.cursor()
        
        # 事件主表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                start_time REAL,
                end_time REAL,
                summary TEXT,
                image_paths TEXT  -- JSON list
            )
        ''')
        
        # 实体表 (Nodes)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                UNIQUE(name, type)
            )
        ''')
        
        # 关系表 (Edges)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER,
                target_id INTEGER,
                relation TEXT,
                event_id TEXT,
                FOREIGN KEY(source_id) REFERENCES entities(id),
                FOREIGN KEY(target_id) REFERENCES entities(id),
                FOREIGN KEY(event_id) REFERENCES events(event_id)
            )
        ''')
        self.sqlite_conn.commit()

    def save_event(self, event_data, summary, kg_data):
        """
        保存事件到双数据库。
        kg_data 是 CognitiveCore 提取出的 {"entities": [], "relationships": []}
        """
        event_id = event_data['event_id']
        start_time = event_data['start_time']
        
        logger.info(f"Saving event {event_id} to Long Term Memory...")
        
        try:
            # --- 1. SQLite 事务开始 ---
            cursor = self.sqlite_conn.cursor()
            
            # 插入事件
            image_paths_json = json.dumps([f['image_path'] for f in event_data['frames']])
            cursor.execute(
                "INSERT OR REPLACE INTO events (event_id, start_time, end_time, summary, image_paths) VALUES (?, ?, ?, ?, ?)",
                (event_id, start_time, event_data['end_time'], summary, image_paths_json)
            )
            
            # 插入实体和关系 (如果 kg_data 存在)
            if kg_data:
                entity_map = {} # name -> db_id
                
                # 先处理所有实体
                for ent in kg_data.get('entities', []):
                    name = ent.get('name', 'Unknown').strip()
                    etype = ent.get('type', 'Object').strip()
                    
                    # 尝试插入，如果存在则忽略
                    cursor.execute("INSERT OR IGNORE INTO entities (name, type) VALUES (?, ?)", (name, etype))
                    
                    # 获取实体ID (无论是刚插入的还是已存在的)
                    cursor.execute("SELECT id FROM entities WHERE name = ? AND type = ?", (name, etype))
                    row = cursor.fetchone()
                    if row:
                        entity_map[name] = row[0]

                # 再处理关系
                for rel in kg_data.get('relationships', []):
                    src_name = rel.get('source')
                    tgt_name = rel.get('target')
                    relation = rel.get('type', 'related_to')
                    
                    src_id = entity_map.get(src_name)
                    tgt_id = entity_map.get(tgt_name)
                    
                    if src_id and tgt_id:
                         cursor.execute(
                            "INSERT INTO relationships (source_id, target_id, relation, event_id) VALUES (?, ?, ?, ?)",
                            (src_id, tgt_id, relation, event_id)
                        )

            self.sqlite_conn.commit()
            # --- SQLite 事务结束 ---

            # --- 2. LanceDB 写入 (在 SQLite 成功后进行) ---
            vector = self.embedding_model.encode(summary)
            self.vector_table.add([{
                "vector": vector,
                "event_id": event_id,
                "timestamp": start_time
            }])
            
            logger.info(f"Event {event_id} saved successfully.")
            return True

        except Exception as e:
            logger.error(f"FAILED to save event {event_id}: {e}", exc_info=True)
            self.sqlite_conn.rollback() # 回滚 SQLite
            return False

    def hybrid_search(self, query_text, top_k=5):
        """
        混合检索：先用向量搜索找到最相关的事件，再从 SQLite 加载这些事件的完整细节。
        """
        # 1. 向量搜索获取候选 Event ID
        query_vector = self.embedding_model.encode(query_text)
        lance_results = self.vector_table.search(query_vector).limit(top_k).to_pandas()
        
        if lance_results.empty:
            return []

        candidate_event_ids = lance_results['event_id'].tolist()
        
        # 2. 从 SQLite 获取完整细节 (包含 KG)
        return self.get_rich_event_details(candidate_event_ids)

    def get_rich_event_details(self, event_ids):
        """
        获取事件的“富”细节，包括摘要、时间和与之关联的知识图谱片段。
        """
        if not event_ids:
            return []

        placeholders = ','.join(['?'] * len(event_ids))
        cursor = self.sqlite_conn.cursor()
        
        # 1. 获取基础事件信息
        cursor.execute(f"SELECT * FROM events WHERE event_id IN ({placeholders}) ORDER BY start_time DESC", event_ids)
        events = [dict(row) for row in cursor.fetchall()]
        
        # 2. 为每个事件填充 KG 细节
        for event in events:
            eid = event['event_id']
            # 查询该事件涉及的所有关系
            cursor.execute('''
                SELECT e1.name as src, r.relation, e2.name as tgt
                FROM relationships r
                JOIN entities e1 ON r.source_id = e1.id
                JOIN entities e2 ON r.target_id = e2.id
                WHERE r.event_id = ?
            ''', (eid,))
            
            relations = cursor.fetchall()
            # 将关系格式化为可读字符串，供 LLM 使用
            kg_text = "; ".join([f"{r['src']} --[{r['relation']}]--> {r['tgt']}" for r in relations])
            event['kg_text'] = kg_text if kg_text else "无明确的结构化关系记录。"
            
        return events

    def get_all_kg_data(self, limit=500):
        """
        获取用于可视化的全局知识图谱数据。
        """
        cursor = self.sqlite_conn.cursor()
        # 获取最新的关系，避免图太大
        cursor.execute(f'''
            SELECT e1.name as source, e1.type as source_type, 
                   r.relation, 
                   e2.name as target, e2.type as target_type,
                   r.event_id
            FROM relationships r
            JOIN entities e1 ON r.source_id = e1.id
            JOIN entities e2 ON r.target_id = e2.id
            ORDER BY r.id DESC
            LIMIT ?
        ''', (limit,))
        return [dict(row) for row in cursor.fetchall()]