# src/memory/long_term_memory.py (最终增强版)
import lancedb
import sqlite3
import pyarrow as pa
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import logging
from openai import OpenAI
import config
import re

logger = logging.getLogger(__name__)

class LongTermMemory:
    def __init__(self, lancedb_path: str, sqlite_path: str):
        # 1. 初始化 LanceDB (向量记忆)
        ldb_path_obj = Path(lancedb_path)
        ldb_path_obj.mkdir(exist_ok=True, parents=True)
        self.vector_db = lancedb.connect(ldb_path_obj)
        
        schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), list_size=384)),
            pa.field("event_id", pa.string()),
            pa.field("summary", pa.string()),
            pa.field("timestamp", pa.float64())
        ])
        self.vector_table = self.vector_db.create_table("semantic_memory", schema=schema, exist_ok=True)
        logger.info(f"LanceDB initialized at {lancedb_path}")

        # 2. 初始化 SQLite (结构化知识图谱)
        Path(sqlite_path).parent.mkdir(exist_ok=True, parents=True)
        self.sqlite_conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        self.sqlite_conn.row_factory = sqlite3.Row
        self._init_sqlite_tables()
        logger.info(f"SQLite Knowledge Graph initialized at {sqlite_path}")

        # 3. 加载嵌入模型
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

        # 4. 初始化LLM客户端，用于NL-to-SQL
        self.llm_client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
        logger.info("LongTermMemory ready.")

    def _init_sqlite_tables(self):
        cursor = self.sqlite_conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY, start_time REAL, end_time REAL,
                summary TEXT, image_paths TEXT, preview_image_path TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, type TEXT NOT NULL, UNIQUE(name, type)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT, source_id INTEGER, target_id INTEGER,
                relation TEXT, event_id TEXT,
                FOREIGN KEY(source_id) REFERENCES entities(id),
                FOREIGN KEY(target_id) REFERENCES entities(id),
                FOREIGN KEY(event_id) REFERENCES events(event_id)
            )
        ''')
        self.sqlite_conn.commit()

    def save_event(self, event_data, summary, kg_data):
        event_id, start_time = event_data['event_id'], event_data['start_time']
        logger.info(f"Saving event {event_id} to LTM...")
        try:
            logger.info(f"Received KG data for saving: {json.dumps(kg_data, indent=2, ensure_ascii=False)}")
            cursor = self.sqlite_conn.cursor()
            image_paths = json.dumps([f['image_path'] for f in event_data['frames']])
            cursor.execute(
                "INSERT OR REPLACE INTO events (event_id, start_time, end_time, summary, image_paths, preview_image_path) VALUES (?, ?, ?, ?, ?, ?)",
                (event_id, start_time, event_data['end_time'], summary, image_paths, event_data.get('preview_image_path'))
            )
            
            if kg_data and 'entities' in kg_data and 'relationships' in kg_data:
                entity_map = {ent['name']: self._get_or_create_entity(cursor, ent['name'], ent['type']) for ent in kg_data['entities']}
                for rel in kg_data['relationships']:
                    src_id, tgt_id = entity_map.get(rel.get('source')), entity_map.get(rel.get('target'))
                    if src_id and tgt_id:
                        cursor.execute("INSERT INTO relationships (source_id, target_id, relation, event_id) VALUES (?, ?, ?, ?)",
                                       (src_id, tgt_id, rel.get('type', 'related_to'), event_id))
            self.sqlite_conn.commit()

            vector = self.embedding_model.encode(summary)
            self.vector_table.add([{"vector": vector, "event_id": event_id, "summary": summary, "timestamp": start_time}])
            
            logger.info(f"Event {event_id} saved successfully.")
            return True
        except Exception as e:
            logger.error(f"FAILED to save event {event_id}: {e}", exc_info=True)
            self.sqlite_conn.rollback()
            return False

    def _get_or_create_entity(self, cursor, name, etype):
        name, etype = (name or 'Unknown').strip(), (etype or 'Object').strip()
        cursor.execute("SELECT id FROM entities WHERE name = ? AND type = ?", (name, etype))
        row = cursor.fetchone()
        if row: return row[0]
        cursor.execute("INSERT INTO entities (name, type) VALUES (?, ?)", (name, etype))
        return cursor.lastrowid

    def semantic_search(self, query_text, top_k=5):
        query_vector = self.embedding_model.encode(query_text)
        results = self.vector_table.search(query_vector).limit(top_k).to_list()
        event_ids = [res['event_id'] for res in results]
        return self.get_rich_event_details(event_ids=event_ids)

    def query_knowledge_graph_by_nl(self, nl_query: str):
        schema_prompt = """
你是一个将自然语言转换为SQL查询的专家。我的数据库有以下三个表：
1. `entities` (id INTEGER, name TEXT, type TEXT) - 存储所有实体，如人、物、地点。
2. `relationships` (id INTEGER, source_id INTEGER, target_id INTEGER, relation TEXT, event_id TEXT) - 存储实体间的关系。
3. `events` (event_id TEXT, summary TEXT, start_time REAL) - 存储事件摘要和时间。

你的任务是根据用户的问题，生成一个可以在这个数据库上执行的SQL查询语句。
- 你必须使用JOIN来连接这些表以回答复杂问题。
- 查询结果应该是有意义的、人类可读的。例如，SELECT e1.name, r.relation, e2.name。
- 只返回SQL查询语句，不要包含任何解释或markdown标记。

用户问题: "{nl_query}"
SQL查询:
"""
        try:
            prompt = schema_prompt.format(nl_query=nl_query)
            response = self.llm_client.chat.completions.create(
                model=config.LLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            sql_query = response.choices[0].message.content.strip()
            sql_query = re.sub(r'```sql\s*|\s*```', '', sql_query)
            
            logger.info(f"Generated SQL for '{nl_query}': {sql_query}")
            
            cursor = self.sqlite_conn.cursor()
            cursor.execute(sql_query)
            results = [dict(row) for row in cursor.fetchall()]
            
            if not results:
                return "知识图谱中没有找到直接相关的答案。"
                
            formatted_result = ""
            for row in results:
                formatted_result += ", ".join([f"{k}: {v}" for k, v in row.items()]) + "\n"
            
            return formatted_result.strip()

        except Exception as e:
            logger.error(f"Failed to query knowledge graph with NL: {e}", exc_info=True)
            return "抱歉，在查询知识图谱时发生了错误。"

    def get_events_for_period(self, start_ts, end_ts):
        cursor = self.sqlite_conn.cursor()
        cursor.execute(
            "SELECT start_time, summary, preview_image_path FROM events WHERE start_time >= ? AND start_time < ? ORDER BY start_time ASC",
            (start_ts, end_ts)
        )
        return [dict(row) for row in cursor.fetchall()]

    # BINGO! 升级此方法以支持两种模式
    def get_rich_event_details(self, event_ids=None, limit=20):
        """
        获取丰富的事件详情。
        - 如果提供了 event_ids 列表，则获取这些特定事件的详情。
        - 如果 event_ids 为 None，则获取最近的 `limit` 个事件。
        """
        cursor = self.sqlite_conn.cursor()
        if event_ids:
            if not event_ids: return []
            placeholders = ','.join(['?'] * len(event_ids))
            cursor.execute(f"SELECT * FROM events WHERE event_id IN ({placeholders}) ORDER BY start_time DESC", event_ids)
        else:
            cursor.execute(f"SELECT * FROM events ORDER BY start_time DESC LIMIT ?", (limit,))
        events = [dict(row) for row in cursor.fetchall()]
        return events

    # BINGO! 正式添加此方法
    def get_kg_for_event(self, event_id: str):
        """获取单个事件关联的知识图谱片段"""
        cursor = self.sqlite_conn.cursor()
        cursor.execute('''
            SELECT e1.name as source, r.relation, e2.name as target 
            FROM relationships r
            JOIN entities e1 ON r.source_id = e1.id 
            JOIN entities e2 ON r.target_id = e2.id
            WHERE r.event_id = ?
        ''', (event_id,))
        results = cursor.fetchall()
        # 返回一个字典列表，方便Pandas处理
        return [{"主语": row['source'], "关系": row['relation'], "宾语": row['target']} for row in results]

    def get_all_kg_data(self, limit=500):
        cursor = self.sqlite_conn.cursor()
        cursor.execute(f'''SELECT e1.name as source, e1.type as source_type, r.relation, 
                                 e2.name as target, e2.type as target_type, r.event_id
                          FROM relationships r
                          JOIN entities e1 ON r.source_id = e1.id JOIN entities e2 ON r.target_id = e2.id
                          ORDER BY r.id DESC LIMIT ?''', (limit,))
        return [dict(row) for row in cursor.fetchall()]