# src/app/agent_tasks.py (最终完整版)
import logging
import os
import sys
from openai import OpenAI
from datetime import datetime, timedelta, date

# --- 动态添加项目根目录到Python路径 ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.memory.long_term_memory import LongTermMemory
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')

class DailyScribeAgent:
    def __init__(self):
        self.memory = LongTermMemory(lancedb_path=config.LANCEDB_PATH, sqlite_path=config.SQLITE_DB_PATH)
        self.llm_client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)

    def _summarize_context(self, context: str, title: str) -> str:
        """
        一个可复用的私有方法，用于将上下文内容交给LLM进行总结。
        :param context: 拼接好的事件记录字符串。
        :param title: 报告的Markdown标题。
        :return: LLM生成的Markdown格式报告。
        """
        prompt = f"""
你是一位专业的分析师和日记作者。请将以下在指定时间段内的活动记录，整合成一篇结构清晰、洞察深刻的Markdown格式分析报告。
你的报告应该：
- 以第三人称来写（例如“在这段时间里，观察对象主要...”）。
- 概括出主要的活动时段和核心内容。
- 文笔自然，像在写一篇生活日志或分析报告。
- 必须使用 `{title}` 作为一级标题。

【活动记录】:
{context}
---
请生成这份Markdown格式的报告。
"""
        try:
            logging.info(f"正在为 '{title}' 调用LLM进行总结...")
            response = self.llm_client.chat.completions.create(
                model=config.LLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"调用LLM为 '{title}' 生成报告失败: {e}", exc_info=True)
            return f"### 生成报告时出错\n\n在调用大模型时发生错误: {e}"

    def generate_daily_summary(self, target_date_str: str = None, return_content: bool = False):
        """
        为【单日】生成每日总结。主要用于自动化的定时任务。
        :param target_date_str: 'YYYY-MM-DD'格式的日期字符串, None表示昨天。
        :param return_content: 如果为True，返回报告内容而不是打印。
        """
        if target_date_str:
            target_d = datetime.strptime(target_date_str, '%Y-%m-%d').date()
        else:
            target_d = (datetime.now() - timedelta(days=1)).date()
        
        report = self.generate_period_summary(start_date=target_d, end_date=target_d)
        
        if return_content:
            return report
        
        # 保持命令行工具的打印功能
        print("\n" + "="*20 + " 每日总结 " + "="*20)
        print(report)
        print("="*52 + "\n")

    def generate_period_summary(self, start_date: date, end_date: date) -> str:
        """
        为【任意时间段】生成总结报告。主要用于UI上的手动报告生成。
        :param start_date: 开始日期 (date对象)
        :param end_date: 结束日期 (date对象)
        :return: Markdown格式的报告字符串。
        """
        start_ts = datetime.combine(start_date, datetime.min.time()).timestamp()
        end_ts = datetime.combine(end_date, datetime.max.time()).timestamp()
        
        date_range_str = f"{start_date.strftime('%Y-%m-%d')}"
        if start_date != end_date:
            date_range_str += f" 至 {end_date.strftime('%Y-%m-%d')}"
            
        logging.info(f"正在为时间段 {date_range_str} 生成总结报告...")
        
        day_memories = self.memory.get_events_for_period(start_ts, end_ts)
        
        title = f"# {date_range_str} 生活报告"

        if not day_memories:
            logging.info("该时间段没有任何记忆记录。")
            return f"{title}\n\n在这段时间内没有记录到任何活动。"

        context = ""
        for row in sorted(day_memories, key=lambda x: x['start_time']):
            # BINGO! 为了在多天报告中区分日期，时间格式包含年月日
            event_time = datetime.fromtimestamp(row['start_time']).strftime('%Y-%m-%d %H:%M')
            context += f"- 时间: {event_time}, 事件: {row['summary']}\n"
            
        return self._summarize_context(context, title)


if __name__ == "__main__":
    # 命令行接口保持不变，用于测试
    scribe = DailyScribeAgent()
    if len(sys.argv) > 1:
        date_arg = sys.argv[1]
        print(f"将为指定日期 {date_arg} 生成总结。")
        scribe.generate_daily_summary(date_arg)
    else:
        print("未指定日期，将默认生成昨天的总结。")
        scribe.generate_daily_summary()
