import lancedb
import sqlite3
import pyarrow as pa
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import logging

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
            cursor = self.sqlite_conn.cursor()
            image_paths = json.dumps([f['image_path'] for f in event_data['frames']])
            cursor.execute(
                "INSERT OR REPLACE INTO events (event_id, start_time, end_time, summary, image_paths, preview_image_path) VALUES (?, ?, ?, ?, ?, ?)",
                (event_id, start_time, event_data['end_time'], summary, image_paths, event_data.get('preview_image_path'))
            )
            
            if kg_data:
                entity_map = {ent['name']: self._get_or_create_entity(cursor, ent['name'], ent['type']) for ent in kg_data.get('entities', [])}
                for rel in kg_data.get('relationships', []):
                    src_id, tgt_id = entity_map.get(rel.get('source')), entity_map.get(rel.get('target'))
                    if src_id and tgt_id:
                        cursor.execute("INSERT INTO relationships (source_id, target_id, relation, event_id) VALUES (?, ?, ?, ?)",
                                       (src_id, tgt_id, rel.get('type', 'related_to'), event_id))
            self.sqlite_conn.commit()

            vector = self.embedding_model.encode(summary)
            self.vector_table.add([{"vector": vector, "event_id": event_id, "timestamp": start_time}])
            
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

    def hybrid_search(self, query_text, top_k=5):
        query_vector = self.embedding_model.encode(query_text)
        lance_results = self.vector_table.search(query_vector).limit(top_k).to_pandas()
        return [] if lance_results.empty else self.get_rich_event_details(lance_results['event_id'].tolist())

    def get_rich_event_details(self, event_ids):
        if not event_ids: return []
        placeholders = ','.join(['?'] * len(event_ids))
        cursor = self.sqlite_conn.cursor()
        cursor.execute(f"SELECT * FROM events WHERE event_id IN ({placeholders}) ORDER BY start_time DESC", event_ids)
        events = [dict(row) for row in cursor.fetchall()]
        for event in events:
            cursor.execute('''SELECT e1.name as src, r.relation, e2.name as tgt FROM relationships r
                              JOIN entities e1 ON r.source_id = e1.id JOIN entities e2 ON r.target_id = e2.id
                              WHERE r.event_id = ?''', (event['event_id'],))
            relations = cursor.fetchall()
            event['kg_text'] = "; ".join([f"{r['src']} --[{r['relation']}]--> {r['tgt']}" for r in relations]) or "No relations found."
        return events

    def get_all_kg_data(self, limit=500):
        cursor = self.sqlite_conn.cursor()
        cursor.execute(f'''SELECT e1.name as source, e1.type as source_type, r.relation, 
                                 e2.name as target, e2.type as target_type, r.event_id
                          FROM relationships r
                          JOIN entities e1 ON r.source_id = e1.id JOIN entities e2 ON r.target_id = e2.id
                          ORDER BY r.id DESC LIMIT ?''', (limit,))
        return [dict(row) for row in cursor.fetchall()]