# debug_kg_extraction.py (一个用于快速测试的独立脚本)

import sys
import os
import sqlite3
import logging
import json

# --- 路径设置 ---
# 确保脚本能找到 src 和 config
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import config
from src.cognition.cognitive_core import CognitiveCore

# --- 日志设置 ---
# 我们希望看到所有模块的详细日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')

def run_test():
    """
    主测试函数
    """
    print("--- 知识图谱提取调试器 ---")
    
    # --- 1. 从数据库获取最新的摘要 ---
    summary_text = None
    try:
        print(f"\n[1/4] 正在连接到数据库: {config.SQLITE_DB_PATH}")
        if not os.path.exists(config.SQLITE_DB_PATH):
            print(f"错误: 数据库文件不存在于 '{config.SQLITE_DB_PATH}'。请先运行主程序生成一些事件。")
            return
            
        conn = sqlite3.connect(config.SQLITE_DB_PATH)
        cursor = conn.cursor()
        
        # 获取最新的一条摘要
        cursor.execute("SELECT summary FROM events ORDER BY start_time DESC LIMIT 1")
        result = cursor.fetchone()
        conn.close()
        
        if result:
            summary_text = result[0]
            print(f"✅ 成功获取到最新的事件摘要:")
            print(f"   '{summary_text}'")
        else:
            print("错误: 数据库中没有找到任何事件。请先运行主程序生成一些事件。")
            return
    except Exception as e:
        print(f"数据库连接或查询失败: {e}")
        return

    # --- 2. 初始化 CognitiveCore ---
    print("\n[2/4] 正在初始化 CognitiveCore...")
    try:
        cognition = CognitiveCore()
        print("✅ CognitiveCore 初始化成功。")
    except Exception as e:
        print(f"CognitiveCore 初始化失败: {e}")
        return

    # --- 3. 调用并测试 _extract_knowledge_graph 函数 ---
    print("\n[3/4] 正在调用知识图谱提取函数...")
    # 注意：在Python中，我们可以通过这种方式直接调用一个类的“私有”方法来进行测试
    kg_data = cognition._extract_knowledge_graph(summary_text)

    # --- 4. 打印最终结果 ---
    print("\n[4/4] 函数执行完毕，这是最终解析出的数据:")
    if kg_data:
        # 使用json.dumps美化输出
        print(json.dumps(kg_data, indent=2, ensure_ascii=False))
    else:
        print("函数返回了 None 或空数据。")
        
    print("\n--- 调试结束 ---")


if __name__ == "__main__":
    run_test()