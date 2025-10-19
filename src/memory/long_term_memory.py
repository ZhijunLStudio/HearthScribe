# src/memory/long_term_memory.py

import lancedb
import pyarrow as pa
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class LongTermMemory:
    def __init__(self, db_path: str):
        db_path = Path(db_path)
        db_path.mkdir(exist_ok=True, parents=True)
        self.db = lancedb.connect(db_path)
        self.table = self._create_table()
        logger.info("正在加载句子嵌入模型 (这可能需要一些时间)...")
        # 使用CPU以获得更好的兼容性
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        logger.info("长期记忆库初始化完成。")

    def _create_table(self):
        # 定义数据库表的结构
        schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), list_size=384)),
            pa.field("event_id", pa.string()),
            pa.field("timestamp", pa.float64()),
            pa.field("summary", pa.string()),
            pa.field("participants", pa.string()),
            pa.field("image_paths", pa.string())
        ])
        try:
            # 尝试创建表，如果已存在则会报错
            tbl = self.db.create_table("memory_events", schema=schema, exist_ok=True)
            logger.info(f"已创建或连接到长期记忆表 'memory_events' at {self.db.uri}")
            return tbl
        except Exception:
             # 如果创建失败（比如因为已存在），则直接打开它
            logger.info(f"已打开现有的长期记忆表 'memory_events' at {self.db.uri}")
            return self.db.open_table("memory_events")

    def save_event_memory(self, event_id, timestamp, summary, participants, image_paths):
        try:
            logger.info(f"正在为事件 {event_id} 生成文本嵌入...")
            vector = self.embedding_model.encode(summary)
            # 确保参与者列表是唯一的，并移除"Unknown"
            unique_participants = list(set(p for p in participants if p != "Unknown"))
            
            self.table.add([{
                "vector": vector,
                "event_id": event_id,
                "timestamp": timestamp,
                "summary": summary,
                "participants": json.dumps(unique_participants),
                "image_paths": json.dumps(image_paths)
            }])
            logger.info(f"事件 {event_id} 已成功存入长期记忆。参与者: {unique_participants}, 摘要: \"{summary[:50]}...\"")
        except Exception as e:
            logger.error(f"保存事件 {event_id} 到长期记忆失败: {e}", exc_info=True)

    def search_memory(self, query: str, top_k=3):
        logger.info(f"正在长期记忆中搜索: '{query}'")
        try:
            query_vector = self.embedding_model.encode(query)
            # BINGO! 核心修正点: .to_df() -> .to_pandas()
            results = self.table.search(query_vector).limit(top_k).to_pandas()
            logger.info(f"搜索完成，找到 {len(results)} 条相关记忆。")
            return results.to_dict('records')
        except Exception as e:
            logger.error(f"搜索记忆失败: {e}", exc_info=True)
            return []