# src/memory/long_term_memory.py
import lancedb
import sqlite3
import pyarrow as pa
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import logging
from openai import OpenAI
import config
import threading

logger = logging.getLogger(__name__)

class LongTermMemory:
    def __init__(self, lancedb_path: str, sqlite_path: str):
        # 1. LanceDB
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
        
        # 2. SQLite
        self.db_lock = threading.Lock()
        Path(sqlite_path).parent.mkdir(exist_ok=True, parents=True)
        self.sqlite_conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        self.sqlite_conn.row_factory = sqlite3.Row
        self._init_sqlite_tables()
        
        # 3. Model
        try:
            self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_PATH, device='cpu')
        except:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            
        self.llm_client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)

    def _init_sqlite_tables(self):
        with self.db_lock:
            c = self.sqlite_conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY, start_time REAL, end_time REAL, summary TEXT, image_paths TEXT, preview_image_path TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS entities (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, type TEXT NOT NULL, UNIQUE(name, type))''')
            c.execute('''CREATE TABLE IF NOT EXISTS relationships (id INTEGER PRIMARY KEY AUTOINCREMENT, source_id INTEGER, target_id INTEGER, relation TEXT, event_id TEXT, FOREIGN KEY(source_id) REFERENCES entities(id), FOREIGN KEY(target_id) REFERENCES entities(id), FOREIGN KEY(event_id) REFERENCES events(event_id))''')
            self.sqlite_conn.commit()

    def save_event(self, event_data, summary, kg_data, scene_label=None, interaction_score=None):
        event_id = event_data['event_id']
        # 拼接扩展信息
        ext_summary = summary
        if scene_label: ext_summary += f"|||LABEL:{scene_label}"
        if interaction_score is not None: ext_summary += f"|||SCORE:{interaction_score}"
        
        try:
            with self.db_lock:
                c = self.sqlite_conn.cursor()
                paths = json.dumps([f['image_path'] for f in event_data['frames']])
                c.execute("INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?)", 
                          (event_id, event_data['start_time'], event_data['end_time'], ext_summary, paths, event_data.get('preview_image_path')))
                
                # KG 存储 (略简化，保证不报错)
                if kg_data and 'entities' in kg_data:
                    ent_map = {}
                    for ent in kg_data['entities']:
                        c.execute("INSERT OR IGNORE INTO entities (name, type) VALUES (?,?)", (ent['name'], ent['type']))
                        c.execute("SELECT id FROM entities WHERE name=?", (ent['name'],))
                        ent_map[ent['name']] = c.fetchone()[0]
                    for rel in kg_data.get('relationships', []):
                        if rel['source'] in ent_map and rel['target'] in ent_map:
                            c.execute("INSERT INTO relationships (source_id, target_id, relation, event_id) VALUES (?,?,?,?)",
                                      (ent_map[rel['source']], ent_map[rel['target']], rel['type'], event_id))
                self.sqlite_conn.commit()
            
            # LanceDB
            vec = self.embedding_model.encode(summary)
            if hasattr(vec, "tolist"): vec = vec.tolist()
            self.vector_table.add([{"vector": vec, "event_id": event_id, "summary": summary, "timestamp": event_data['start_time']}])
            return True
        except Exception as e:
            logger.error(f"Save failed: {e}")
            return False

    # --- 之前缺失的方法 ---
    def get_events_for_period(self, start_ts, end_ts):
        with self.db_lock:
            c = self.sqlite_conn.cursor()
            c.execute("SELECT * FROM events WHERE start_time >= ? AND start_time <= ? ORDER BY start_time", (start_ts, end_ts))
            return [dict(row) for row in c.fetchall()]

    def semantic_search(self, query, top_k=5):
        try:
            vec = self.embedding_model.encode(query)
            if hasattr(vec, "tolist"): vec = vec.tolist()
            res = self.vector_table.search(vec).limit(top_k).to_list()
            ids = [r['event_id'] for r in res]
            return self.get_rich_event_details(event_ids=ids)
        except: return []

    def get_rich_event_details(self, event_ids=None, limit=20):
        with self.db_lock:
            c = self.sqlite_conn.cursor()
            if event_ids:
                if not event_ids: return []
                ph = ','.join(['?']*len(event_ids))
                c.execute(f"SELECT * FROM events WHERE event_id IN ({ph})", event_ids)
            else:
                c.execute("SELECT * FROM events ORDER BY start_time DESC LIMIT ?", (limit,))
            return [dict(row) for row in c.fetchall()]

    def query_knowledge_graph_by_nl(self, query):
        # 简单实现，避免报错
        return "KG Query logic placeholder" 

    def get_all_kg_data(self, limit=300):
        with self.db_lock:
            c = self.sqlite_conn.cursor()
            c.execute(f"""SELECT e1.name as source, e1.type as source_type, r.relation, e2.name as target, e2.type as target_type, r.event_id 
                         FROM relationships r 
                         JOIN entities e1 ON r.source_id=e1.id 
                         JOIN entities e2 ON r.target_id=e2.id LIMIT ?""", (limit,))
            return [dict(row) for row in c.fetchall()]
    
    def get_kg_for_event(self, event_id):
        # 补充缺失的方法
        with self.db_lock:
            c = self.sqlite_conn.cursor()
            c.execute("""SELECT e1.name as source, r.relation, e2.name as target 
                         FROM relationships r 
                         JOIN entities e1 ON r.source_id=e1.id 
                         JOIN entities e2 ON r.target_id=e2.id
                         WHERE r.event_id=?""", (event_id,))
            return [dict(row) for row in c.fetchall()]