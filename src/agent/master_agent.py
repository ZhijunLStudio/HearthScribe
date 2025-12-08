import logging
from openai import OpenAI
import config
from src.memory.long_term_memory import LongTermMemory
from datetime import datetime
import json
import re

logger = logging.getLogger(__name__)

class MasterAgent:
    def __init__(self, memory: LongTermMemory):
        self.memory = memory
        # 兼容 config.API_KEY 和 config.ERNIE_API_KEY
        api_key = getattr(config, 'API_KEY', getattr(config, 'ERNIE_API_KEY', ''))
        base_url = getattr(config, 'BASE_URL', getattr(config, 'ERNIE_BASE_URL', ''))
        
        self.llm_client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info("MasterAgent initialized.")

    def _get_query_route(self, query: str) -> str:
        prompt = f"""
        任务：路由分类。选项：
        - memory_retrieval: 具体事件回忆 (钥匙在哪? 刚才发生了什么?)
        - graph_reasoning: 关系/统计 (我和谁出现过? 谁最常来?)
        - summarization: 时间段总结 (今天上午干了什么?)
        用户问题: "{query}"
        只返回分类名称。
        """
        try:
            model_name = getattr(config, 'AI_THINKING_MODEL', 'ernie-4.5-vl-28b-a3b-thinking')
            response = self.llm_client.chat.completions.create(
                model=model_name, 
                messages=[{"role": "user", "content": prompt}], 
                temperature=0.0
            )
            route = response.choices[0].message.content.strip().lower()
            return route if route in ["memory_retrieval", "graph_reasoning", "summarization"] else "memory_retrieval"
        except Exception as e:
            logger.error(f"Routing error: {e}")
            return "memory_retrieval"

    def _memory_retrieval_expert(self, query: str, streamer):
        yield {"status": "retrieving", "content": "🔍 正在检索记忆..."}
        
        # 优先使用 semantic_search，如果没有则降级
        if hasattr(self.memory, 'semantic_search'):
            results = self.memory.semantic_search(query, top_k=5)
        else:
            results = self.memory.get_rich_event_details(limit=5)
            
        if not results:
            yield None
            return

        context_str = ""
        for i, event in enumerate(results):
            t = datetime.fromtimestamp(event['start_time']).strftime('%Y-%m-%d %H:%M:%S')
            context_str += f"[{t}] {event['summary']}\n"
            
        yield (context_str, results)

    def _graph_reasoning_expert(self, query: str, streamer):
        yield {"status": "retrieving", "content": "🕸️ 正在查询知识图谱..."}
        res = self.memory.query_knowledge_graph_by_nl(query)
        yield (f"知识图谱查询结果:\n{res}", None)

    def _summarization_expert(self, query: str, streamer):
        yield {"status": "retrieving", "content": "📚 正在聚合今日活动..."}
        today_start = datetime.now().replace(hour=0, minute=0, second=0).timestamp()
        events = self.memory.get_events_for_period(today_start, datetime.now().timestamp())
        
        if not events:
            yield ("今日暂无记录。", None)
            return
            
        context = "\n".join([f"[{datetime.fromtimestamp(e['start_time']).strftime('%H:%M')}] {e['summary']}" for e in events])
        yield (context, None)

    def _generate_final_answer(self, query, context):
        prompt = f"""
        基于以下记忆回答问题。
        记忆:
        {context}
        问题: {query}
        """
        try:
            model_name = getattr(config, 'AI_THINKING_MODEL', 'ernie-4.5-vl-28b-a3b-thinking')
            resp = self.llm_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            for chunk in resp:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"生成出错: {e}"

    def execute_query_steps(self, query, streamer):
        route = self._get_query_route(query)
        yield {"status": "routing", "content": f"决策路径: `{route}`"}
        
        expert_gen = None
        if route == 'memory_retrieval': expert_gen = self._memory_retrieval_expert(query, streamer)
        elif route == 'graph_reasoning': expert_gen = self._graph_reasoning_expert(query, streamer)
        elif route == 'summarization': expert_gen = self._summarization_expert(query, streamer)
        
        context, evidence = None, None
        if expert_gen:
            for item in expert_gen:
                if isinstance(item, dict): yield item
                elif isinstance(item, tuple): context, evidence = item
        
        if not context:
            yield {"status": "done", "content": "我没有找到相关记忆。"}
            return
            
        yield {"status": "generating", "content": ""}
        full_ans = ""
        for chunk in self._generate_final_answer(query, context):
            full_ans += chunk
            yield {"status": "generating", "content": full_ans}
            
        yield {"status": "done", "content": full_ans}