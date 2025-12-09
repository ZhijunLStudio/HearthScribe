# src/app/qa_interface.py
import logging
import os
import sys
from openai import OpenAI
from datetime import datetime
import json

# --- 动态添加项目根目录到Python路径 ---
# 这使得无论您在哪个目录下运行此脚本，它都能找到src和config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.memory.long_term_memory import LongTermMemory
import config

# 配置一个简单的日志，只输出到控制台
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QA_Agent:
    def __init__(self):
        print("\n正在初始化命令行问答助手...")
        self.memory = LongTermMemory(db_path=config.SQLITE_DB_PATH)
        self.llm_client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
        print("\n问答助手已就绪！请输入您的问题，或输入 '退出' 来结束。")

    def answer(self, user_query: str):
        # 1. 向量检索
        retrieved_memories = self.memory.search_memory(user_query, top_k=3)

        if not retrieved_memories:
            print("\n抱歉，我在记忆中没有找到相关信息。\n")
            return

        # 2. 构建上下文
        context = "这是我记忆中的一些相关片段，请基于这些信息来回答用户的问题。\n\n"
        image_evidence = []
        for i, mem in enumerate(retrieved_memories):
            event_time = datetime.fromtimestamp(mem['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            context += f"--- 记忆片段 {i+1} (发生于 {event_time}) ---\n"
            context += f"摘要: {mem['summary']}\n"
            context += f"参与者: {json.loads(mem.get('participants', '[]'))}\n\n"
            
            image_paths = json.loads(mem.get('image_paths', '[]'))
            if image_paths:
                image_evidence.append(image_paths[0])

        # 3. 调用LLM生成答案
        prompt = f"""
{context}
---
用户的问题是: "{user_query}"

请根据上面的背景记忆信息，用第一人称（“我记得...”）和友好的口吻，自然地回答用户的问题。如果信息不足，可以说“我只记得...，更具体的细节我没有注意到。”
"""
        try:
            print("\n正在思考，请稍候...")
            response = self.llm_client.chat.completions.create(
                model=config.LLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            answer_text = response.choices[0].message.content
            
            print("\n-------------------- 回答 --------------------")
            print(answer_text)
            if image_evidence:
                print("\n相关记忆图片证据:")
                for path in set(image_evidence): # 去重
                    print(f"- {path}")
            print("------------------------------------------\n")

        except Exception as e:
            logging.error(f"调用LLM生成答案失败: {e}")


def start_qa_session():
    agent = QA_Agent()
    while True:
        try:
            query = input("您想问什么？> ")
            if query.lower() in ['退出', 'exit', 'quit']:
                break
            if not query.strip():
                continue
            agent.answer(query)
        except (KeyboardInterrupt, EOFError):
            break
    print("\n感谢使用，再见！")

if __name__ == "__main__":
    start_qa_session()
