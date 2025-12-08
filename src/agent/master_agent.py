import logging
from openai import OpenAI
import config
from src.memory.long_term_memory import LongTermMemory
from datetime import datetime
import json
import traceback

logger = logging.getLogger(__name__)

class MasterAgent:
    def __init__(self, memory: LongTermMemory):
        self.memory = memory
        # 兼容 config 写法，优先读取 ERNIE 配置
        api_key = getattr(config, 'API_KEY', getattr(config, 'ERNIE_API_KEY', ''))
        base_url = getattr(config, 'BASE_URL', getattr(config, 'ERNIE_BASE_URL', ''))
        
        self.llm_client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info("MasterAgent initialized.")

    def _get_query_route(self, query: str) -> str:
        """
        简单路由：判断用户是想查记忆、查图谱还是做总结。
        """
        prompt = f"""
        任务：意图分类。
        选项：
        - memory_retrieval: 查询具体事件、找人、找东西 (例如: "张三什么时候回来的?", "谁在客厅?")
        - graph_reasoning: 查询统计、关系、频率 (例如: "我和谁互动最多?", "谁最常来?")
        - summarization: 时间段总结 (例如: "今天上午发生了什么?", "生成日报")
        
        用户问题: "{query}"
        
        只返回分类名称，不要标点符号。
        """
        try:
            # 使用思考模型进行路由决策
            model_name = getattr(config, 'AI_THINKING_MODEL', 'ernie-4.5-vl-28b-a3b-thinking')
            response = self.llm_client.chat.completions.create(
                model=model_name, 
                messages=[{"role": "user", "content": prompt}], 
                temperature=0.0
            )
            route = response.choices[0].message.content.strip().lower()
            valid_routes = ["memory_retrieval", "graph_reasoning", "summarization"]
            return route if route in valid_routes else "memory_retrieval"
        except Exception as e:
            logger.error(f"Routing error: {e}")
            return "memory_retrieval"

    def _memory_retrieval_expert(self, query: str):
        """记忆检索专家：返回检索到的 Context 字符串"""
        # 1. 尝试语义搜索
        if hasattr(self.memory, 'semantic_search'):
            # 搜索最相关的 5 条
            results = self.memory.semantic_search(query, top_k=5)
        else:
            # 降级方案
            results = self.memory.get_rich_event_details(limit=5)
            
        if not results:
            return None, None

        # 2. 构建上下文
        context_str = "【相关记忆片段】:\n"
        for i, event in enumerate(results):
            t = datetime.fromtimestamp(event['start_time']).strftime('%Y-%m-%d %H:%M:%S')
            # 确保解析摘要中的标签和评分
            summary_text = event['summary'].split("|||")[0] 
            context_str += f"- 时间: {t} | 事件: {summary_text}\n"
            
        return context_str, results

    def _graph_reasoning_expert(self, query: str):
        """知识图谱专家"""
        res = self.memory.query_knowledge_graph_by_nl(query)
        context_str = f"【知识图谱数据】:\n{res}"
        return context_str, None

    def _summarization_expert(self, query: str):
        """总结专家"""
        today_start = datetime.now().replace(hour=0, minute=0, second=0).timestamp()
        events = self.memory.get_events_for_period(today_start, datetime.now().timestamp())
        
        if not events:
            return "今日暂无记录。", None
            
        context_str = "【今日活动流水】:\n"
        for e in events:
            t = datetime.fromtimestamp(e['start_time']).strftime('%H:%M')
            summary_text = e['summary'].split("|||")[0]
            context_str += f"- [{t}] {summary_text}\n"
            
        return context_str, None

    def _generate_final_answer(self, query, context):
        """生成最终回答 (流式)"""
        prompt = f"""
        你是一个智能家庭管家。请根据以下提供的【记忆信息】来回答用户的【问题】。
        
        要求：
        1. 必须基于提供的记忆信息回答，不要编造。
        2. 引用具体的时间点（例如“在14:30分的时候...”）。
        3. 如果记忆中包含具体人名（如张三、李四），请明确指出，不要只说“有人”。
        4. 语气自然、亲切。
        
        {context}
        
        【用户问题】: {query}
        """
        try:
            model_name = getattr(config, 'AI_THINKING_MODEL', 'ernie-4.5-vl-28b-a3b-thinking')
            resp = self.llm_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            for chunk in resp:
                # --- 关键修复：防止 list index out of range ---
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content
        except Exception as e:
            logger.error(f"Generate answer error: {e}")
            yield f" [生成回答时发生错误: {str(e)}]"

    def execute_query_steps(self, query, streamer=None):
        """
        主执行入口，返回生成器。
        Yields:
            {'status': 'thinking', 'content': '...'}  -> 用于前端显示思考过程
            {'status': 'answer', 'content': '...'}    -> 用于前端显示最终回答
        """
        # Step 1: 路由决策
        yield {"status": "thinking", "content": "🤔 正在分析您的问题意图..."}
        route = self._get_query_route(query)
        yield {"status": "thinking", "content": f"👉 决策路径: `{route}`"}
        
        # Step 2: 调用专家检索
        expert_gen = None
        context = None
        
        if route == 'memory_retrieval':
            yield {"status": "thinking", "content": "🔍 正在检索语义记忆库..."}
            context, _ = self._memory_retrieval_expert(query)
        elif route == 'graph_reasoning':
            yield {"status": "thinking", "content": "🕸️ 正在查询知识图谱..."}
            context, _ = self._graph_reasoning_expert(query)
        elif route == 'summarization':
            yield {"status": "thinking", "content": "📅 正在聚合今日活动记录..."}
            context, _ = self._summarization_expert(query)
        
        if not context:
            yield {"status": "thinking", "content": "❌ 未找到相关信息。"}
            yield {"status": "answer", "content": "抱歉，我在记忆中没有找到相关的信息。"}
            return

        yield {"status": "thinking", "content": "✅ 信息检索完毕，正在组织语言..."}
        
        # Step 3: 生成回答
        full_ans = ""
        for chunk in self._generate_final_answer(query, context):
            full_ans += chunk
            # 实时流式输出
            yield {"status": "answer", "content": chunk}