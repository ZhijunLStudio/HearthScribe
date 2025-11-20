import os
import sys
import logging
from datetime import datetime

# --- 路径设置 ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.memory.long_term_memory import LongTermMemory
from src.agent.master_agent import MasterAgent
import config

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(name)s] - %(levelname)s: %(message)s')
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

def run_test(query):
    """运行一次完整的问答测试"""
    print("\n" + "="*80)
    print(f"🤔 开始测试查询: '{query}'")
    print("="*80)

    try:
        # 初始化模块
        memory = LongTermMemory(config.LANCEDB_PATH, config.SQLITE_DB_PATH)
        agent = MasterAgent(memory)
    except Exception as e:
        print(f"❌ 初始化模块失败: {e}")
        return

    # 定义一个简单的streamer回调函数，用于在终端打印Agent的思考过程
    def terminal_streamer(content):
        print(f"🧠 [Agent চিন্তা]: {content.strip()}")

    # 执行查询
    step_generator = agent.execute_query_steps(query, terminal_streamer)
    
    final_step = None
    try:
        for step in step_generator:
            final_step = step
    except Exception as e:
        print(f"\n💥 Agent执行过程中发生错误: {e}")
        return

    # --- 打印最终结果 ---
    print("\n" + "-"*80)
    print("✅ 测试执行完毕。最终结果如下：")
    
    if final_step and isinstance(final_step, dict):
        answer = final_step.get('content', '没有获取到最终回答。')
        evidence = final_step.get('evidence')

        print("\n🤖 最终回答:")
        print(f"> {answer}")

        if evidence:
            print("\n📚 依据的记忆证据:")
            unique_evidence = {ev['event_id']: ev for ev in evidence}.values()
            for ev in unique_evidence:
                time_str = datetime.fromtimestamp(ev['start_time']).strftime('%Y-%m-%d %H:%M')
                print(f"- [{time_str}] {ev['summary']}")
        else:
            print("\n- 未找到相关记忆证据。")

    else:
        print("\n- 未能获取到结构化的最终结果。")
    
    print("-" * 80)


if __name__ == "__main__":
    # --- 在这里输入您想测试的问题 ---
    test_query = "lizhijun在做什么"
    
    run_test(test_query)
    
    # 您也可以添加更多测试用例
    # run_test("lizhijun戴着眼镜吗")