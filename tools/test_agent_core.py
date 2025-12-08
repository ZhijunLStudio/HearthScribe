import os
import sys
import time
from datetime import datetime
import traceback

# --- 1. 关键修复：路径配置 ---
# 获取当前脚本所在目录 (tools/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (tools 的上一级)
project_root = os.path.dirname(current_dir)
# 将根目录加入 Python 搜索路径，这样才能 import config 和 src
sys.path.insert(0, project_root)

# print(f"调试: 项目根目录设为 -> {project_root}")

try:
    import config
    from src.memory.long_term_memory import LongTermMemory
    from src.agent.master_agent import MasterAgent
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    print(f"请检查目录结构。期望 config.py 位于: {os.path.join(project_root, 'config.py')}")
    sys.exit(1)

# --- 颜色代码 ---
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
RESET = "\033[0m"

def print_step(msg):
    print(f"\n{YELLOW}=== {msg} ==={RESET}")

def test_config():
    print_step("1. 检查配置与环境")
    
    # 优先读取 ERNIE 配置
    api_key = getattr(config, 'API_KEY', getattr(config, 'ERNIE_API_KEY', None))
    base_url = getattr(config, 'BASE_URL', getattr(config, 'ERNIE_BASE_URL', None))
    
    if api_key:
        masked_key = api_key[:4] + "****" + api_key[-4:]
        print(f"✅ API Key 已加载: {masked_key}")
    else:
        print(f"{RED}❌ 未检测到 API_KEY！请检查 config.py 或 .env{RESET}")
        return False
        
    print(f"✅ Base URL: {base_url}")
    # 打印模型配置
    vl_model = getattr(config, 'AI_VL_MODEL', '未定义')
    think_model = getattr(config, 'AI_THINKING_MODEL', '未定义')
    print(f"✅ 视觉模型: {vl_model}")
    print(f"✅ 思考模型: {think_model}")
    
    return True

def test_database():
    print_step("2. 检查数据库连接与数据")
    
    try:
        # 初始化记忆模块
        memory = LongTermMemory(config.LANCEDB_PATH, config.SQLITE_DB_PATH)
        
        # 简单查一下 SQLite
        with memory.db_lock:
            c = memory.sqlite_conn.cursor()
            c.execute("SELECT COUNT(*) FROM events")
            count = c.fetchone()[0]
            
        print(f"✅ 数据库连接成功")
        if count > 0:
            print(f"✅ 数据库中已有 {GREEN}{count}{RESET} 条记忆片段。")
            
            # 查一条最新的看看
            c.execute("SELECT summary, start_time FROM events ORDER BY start_time DESC LIMIT 1")
            row = c.fetchone()
            if row:
                t_str = datetime.fromtimestamp(row[1]).strftime('%H:%M:%S')
                # 简单清洗一下摘要显示
                summary_preview = row[0].split("|||")[0][:50]
                print(f"   📝 最新一条 ({t_str}): {summary_preview}...")
        else:
            print(f"{YELLOW}⚠️ 警告: 数据库是空的！{RESET}")
            print("   (Agent 可能会回答“不知道”，这是正常的。请运行 main.py 捕捉画面后再试)")
            
        return memory
    except Exception as e:
        print(f"{RED}❌ 数据库连接失败: {e}{RESET}")
        traceback.print_exc()
        return None

def test_agent_interaction(memory):
    print_step("3. 测试 Agent 思考与回答 (流式)")
    
    try:
        agent = MasterAgent(memory)
        query = "今天发生了什么？" 
        
        print(f"👤 User: {query}\n")
        print(f"{CYAN}[AI 开始响应...]{RESET}")
        
        # 获取生成器
        gen = agent.execute_query_steps(query)
        
        full_answer = ""
        has_thinking = False
        
        # 模拟前端的循环
        for step in gen:
            # 检查数据结构
            if not isinstance(step, dict):
                print(f"{RED}❌ 格式错误: Agent 返回的不是 dict, 而是 {type(step)}{RESET}")
                continue
                
            status = step.get('status')
            content = step.get('content')
            
            if status == "thinking":
                has_thinking = True
                # 打印思考过程
                print(f"{YELLOW}[思考] {content}{RESET}")
            elif status == "answer":
                # 打印最终答案 (流式不换行)
                print(f"{GREEN}{content}{RESET}", end="", flush=True)
                full_answer += content
                
        print(f"\n\n{CYAN}[响应结束]{RESET}")
        
        if not has_thinking:
             print(f"{RED}❌ 失败: 没有接收到任何思考过程状态 (thinking)。{RESET}")
        elif not full_answer:
            print(f"{RED}❌ 失败: 最终回答为空！可能是大模型调用超时或出错。{RESET}")
        else:
            print(f"✅ 测试通过！Agent 工作正常。")
            
    except Exception as e:
        print(f"\n{RED}❌ Agent 运行崩溃: {e}{RESET}")
        traceback.print_exc()

if __name__ == "__main__":
    print(f"运行目录: {os.getcwd()}")
    if test_config():
        mem_instance = test_database()
        if mem_instance:
            test_agent_interaction(mem_instance)