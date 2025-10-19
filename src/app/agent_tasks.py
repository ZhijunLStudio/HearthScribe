# src/app/agent_tasks.py
import logging
import os
import sys
from openai import OpenAI
from datetime import datetime, timedelta
import json
import pandas as pd

# --- 动态添加项目根目录到Python路径 ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.memory.long_term_memory import LongTermMemory
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')

class DailyScribeAgent:
    def __init__(self):
        self.memory = LongTermMemory(db_path=config.DB_PATH)
        self.llm_client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)

    def generate_daily_summary(self, target_date_str: str = None):
        """
        生成指定日期的每日总结。
        :param target_date_str: 'YYYY-MM-DD'格式的日期字符串, None表示昨天
        """
        if target_date_str:
            try:
                target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
            except ValueError:
                logging.error("日期格式错误，请输入 'YYYY-MM-DD' 格式。")
                return
        else:
            target_date = datetime.now() - timedelta(days=1)
        
        start_of_day = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0)
        end_of_day = start_of_day + timedelta(days=1)
        
        start_ts = start_of_day.timestamp()
        end_ts = end_of_day.timestamp()

        logging.info(f"正在为日期 {start_of_day.strftime('%Y-%m-%d')} 生成每日总结...")
        
        try:
            all_memories_df = self.memory.table.to_pandas_df()
        except Exception as e:
            logging.error(f"从数据库读取记忆失败: {e}")
            return
            
        day_memories = all_memories_df[(all_memories_df['timestamp'] >= start_ts) & (all_memories_df['timestamp'] < end_ts)]
        
        if day_memories.empty:
            logging.info("该日期没有任何记忆记录。")
            return

        context = f"以下是 {start_of_day.strftime('%Y-%m-%d')} 这一天中，按时间顺序记录的所有活动片段。\n\n"
        for _, row in day_memories.sort_values(by='timestamp').iterrows():
            event_time = datetime.fromtimestamp(row['timestamp']).strftime('%H:%M')
            participants = json.loads(row.get('participants', '[]'))
            context += f"- 时间: {event_time}, 人物: {participants or '未知'}, 事件: {row['summary']}\n"

        prompt = f"""
你是一位细心的日记作者。请将以下一天中零散的活动记录，整合成一篇通顺、连贯的每日总结。
你的总结应该：
- 以第三人称（例如“今天，lizhijun主要...”）来写。
- 概括出主要的活动时段和内容。
- 文笔自然，像在写一篇生活日志。

【{start_of_day.strftime('%Y-%m-%d')} 的活动记录】:
{context}
---
请生成这篇每日总结。
"""
        try:
            logging.info("正在调用LLM进行总结...")
            response = self.llm_client.chat.completions.create(
                model=config.LLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6
            )
            summary_text = response.choices[0].message.content
            
            print("\n" + "="*20 + " 每日总结 " + "="*20)
            print(f"日期: {start_of_day.strftime('%Y-%m-%d')}")
            print("-" * 52)
            print(summary_text)
            print("="*52 + "\n")
            # 在实际应用中，这里可以将总结存入数据库的新表
            
        except Exception as e:
            logging.error(f"调用LLM生成每日总结失败: {e}")

if __name__ == "__main__":
    scribe = DailyScribeAgent()
    # 允许从命令行传入日期参数
    if len(sys.argv) > 1:
        date_arg = sys.argv[1]
        print(f"将为指定日期 {date_arg} 生成总结。")
        scribe.generate_daily_summary(date_arg)
    else:
        print("未指定日期，将默认生成昨天的总结。")
        scribe.generate_daily_summary()