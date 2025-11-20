import sqlite3
import pandas as pd
import os

# --- 配置 ---
SQLITE_DB_PATH = "./memory_db/knowledge.db"

def inspect_memories():
    if not os.path.exists(SQLITE_DB_PATH):
        print(f"错误：数据库文件 '{SQLITE_DB_PATH}' 不存在。")
        return

    print(f"--- 正在查询数据库: {SQLITE_DB_PATH} ---")
    conn = sqlite3.connect(SQLITE_DB_PATH)
    try:
        # 使用pandas可以更漂亮地显示结果
        df = pd.read_sql_query("SELECT event_id, summary FROM events ORDER BY start_time DESC LIMIT 10", conn)
        print("最近的10条记忆摘要:")
        pd.set_option('display.max_colwidth', None) # 显示完整的摘要内容
        print(df)
    except Exception as e:
        print(f"查询失败: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    inspect_memories()