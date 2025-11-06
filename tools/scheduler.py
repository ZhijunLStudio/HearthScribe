# tools/scheduler.py (全新文件)
import schedule
import time
import logging
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from src.app.agent_tasks import DailyScribeAgent # 我们将复用/改造这个类

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

def generate_daily_report_job():
    """定时任务的具体执行逻辑"""
    today_str = datetime.now().strftime('%Y-%m-%d')
    logging.info(f"--- 🚀 定时任务触发: 开始为 {today_str} 生成每日报告 ---")
    
    try:
        # 确保报告目录存在
        reports_dir = Path(config.DAILY_REPORTS_PATH)
        reports_dir.mkdir(exist_ok=True)
        
        # 实例化报告生成Agent
        scribe = DailyScribeAgent()
        
        # 调用生成方法，并获取报告内容
        report_content = scribe.generate_daily_summary(target_date_str=today_str, return_content=True)
        
        if report_content:
            report_file_path = reports_dir / f"report_{today_str}.md"
            with open(report_file_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logging.info(f"✅ 报告已成功生成并保存到: {report_file_path}")
        else:
            logging.warning(f"🟡 未能生成报告，可能今天没有活动记录。")

    except Exception as e:
        logging.error(f"💥 生成每日报告时发生严重错误: {e}", exc_info=True)

def main():
    setup_logging()
    logging.info("--- 自动化报告调度器已启动 ---")
    logging.info("将在每天的 22:00 执行每日报告生成任务。")

    # 定义调度规则
    schedule.every().day.at("22:00").do(generate_daily_report_job)
    
    # 立即执行一次用于测试 (可选)
    # logging.info("为了测试，将立即执行一次任务...")
    # generate_daily_report_job()

    while True:
        schedule.run_pending()
        time.sleep(60) # 每分钟检查一次任务

if __name__ == "__main__":
    main()