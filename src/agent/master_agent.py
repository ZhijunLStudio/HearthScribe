# src/agent/master_agent.py (全新文件)
import logging
from openai import OpenAI
import config
from src.memory.long_term_memory import LongTermMemory
from datetime import datetime, timedelta
import re
import json

logger = logging.getLogger(__name__)

class MasterAgent:
    def __init__(self, memory: LongTermMemory):
        self.memory = memory
        self.llm_client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
        logger.info("MasterAgent initialized.")

    def execute_query(self, query: str, full_response_streamer):
        """主执行方法，处理用户查询并返回生成器"""
        
        # 1. 查询路由 (Query Routing)
        try:
            route = self._get_query_route(query)
            logger.info(f"Query '{query}' routed to: {route}")
            yield f"**决策路径**: `{route}`\n\n---\n\n"
        except Exception as e:
            logger.error(f"Query routing failed: {e}", exc_info=True)
            yield "抱歉，我在决策如何处理您的问题时遇到了困难。"
            return

        # 2. 根据路由调用专家 (Expert Invocation)
        context = ""
        evidence = []
        if route == "memory_retrieval":
            # 2a. 记忆检索专家 (Corrective RAG - Query Refinement)
            context, evidence = self._memory_retrieval_expert(query, full_response_streamer)
        elif route == "graph_reasoning":
            # 2b. 知识图谱推理专家 (Graph RAG)
            context = self._graph_reasoning_expert(query, full_response_streamer)
        elif route == "summarization":
            # 2c. 总结专家
            context = self._summarization_expert(query, full_response_streamer)
        else:
            # 默认或未知路由
            context, evidence = self._memory_retrieval_expert(query, full_response_streamer)

        if not context:
            yield "抱歉，根据您的问题，我没有在我的知识库中找到相关信息。"
            return

        # 3. 最终答案生成 (Final Answer Generation)
        system_prompt = "你是一个AI Agent的记忆核心。请基于以下提供的【背景上下文】，用自然、流畅的口吻回答用户的问题。如果信息不足，就坦诚说明。"
        user_prompt = f"【背景上下文】\n{context}\n\n【用户问题】\n{query}"
        
        try:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            response = self.llm_client.chat.completions.create(model=config.LLM_MODEL_NAME, messages=messages, stream=True, temperature=0.5)
            
            # 在返回答案前，先返回证据
            if evidence:
                yield "\n\n---\n\n**相关记忆证据:**\n"
                for ev in evidence:
                    preview_path = ev.get('preview_image_path', '')
                    summary = ev.get('summary', '无摘要')
                    event_time = datetime.fromtimestamp(ev['start_time']).strftime('%Y-%m-%d %H:%M')
                    # Streamlit Markdown支持图片: f"![{alt_text}]({image_url})"
                    yield f"- **[{event_time}]** {summary}\n"
                    # 这里假设Streamlit可以访问到这个路径。在实际部署中可能需要配置静态文件服务。
                    if preview_path:
                         yield f"  - *预览图路径: `{preview_path}`*\n"


            yield "\n\n---\n\n**综合回答:**\n\n"
            for chunk in response:
                if chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Final answer generation failed: {e}", exc_info=True)
            yield "抱歉，我在生成最终答案时遇到了错误。"


    def _get_query_route(self, query: str) -> str:
        prompt = f"""
你是一个任务分析与路由专家。根据用户的问题，判断最适合处理该问题的专家。
可用专家如下：
- `memory_retrieval`: 当问题是关于具体的、近期发生的事件，需要回忆某个场景或动作时。例如："我刚才把钥匙放哪了？", "lizhijun下午在做什么？"
- `graph_reasoning`: 当问题是关于实体之间的关系、规律、频率或统计时，需要进行逻辑推理。例如："我和谁一起出现过？", "我的杯子通常放在哪里？", "lizhijun最常在哪个房间活动？"
- `summarization`: 当问题要求对一段时间的活动进行总结或回顾时。例如："总结我今天上午的活动", "上周我都干了些什么？"

用户问题: "{query}"

请只返回最合适的专家名称 (memory_retrieval, graph_reasoning, summarization)。
"""
        response = self.llm_client.chat.completions.create(
            model=config.LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        route = response.choices[0].message.content.strip().lower()
        
        # 基本的验证，确保返回的是我们定义的路由之一
        valid_routes = ["memory_retrieval", "graph_reasoning", "summarization"]
        if route in valid_routes:
            return route
        return "memory_retrieval" # 默认为记忆检索


    def _memory_retrieval_expert(self, query: str, streamer):
        streamer("**调用专家**: `记忆检索专家`\n")
        
        # Corrective RAG - Query Refinement
        refined_query = self._refine_query_for_retrieval(query)
        streamer(f"**优化查询**: `{query}` -> `{refined_query}`\n")

        retrieved_events = self.memory.semantic_search(refined_query, top_k=4)
        
        if not retrieved_events:
            return None, None
        
        context_str = "以下是我检索到的相关记忆片段：\n\n"
        for i, event in enumerate(retrieved_events):
            time_str = datetime.fromtimestamp(event['start_time']).strftime('%Y-%m-%d %H:%M:%S')
            context_str += f"--- 记忆片段 {i+1} [{time_str}] ---\n"
            context_str += f"摘要: {event['summary']}\n\n"
            
        return context_str, retrieved_events

    def _refine_query_for_retrieval(self, query: str) -> str:
        prompt = f"""
你是一个查询优化专家。用户的原始问题可能很口语化。请将其改写成一个或多个更适合向量数据库进行语义搜索的陈述句或关键词组，用空格隔开。

原始问题: "{query}"

优化后的查询:
"""
        response = self.llm_client.chat.completions.create(
            model=config.LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()

    def _graph_reasoning_expert(self, query: str, streamer):
        streamer("**调用专家**: `知识图谱推理专家`\n")
        streamer("**行动**: `自然语言转SQL查询`\n")
        
        result = self.memory.query_knowledge_graph_by_nl(query)
        
        context_str = f"以下是我从知识图谱中查询到的信息：\n\n{result}"
        return context_str
        
    def _summarization_expert(self, query: str, streamer):
        streamer("**调用专家**: `活动总结专家`\n")
        
        # 1. 提取时间范围
        start_ts, end_ts = self._extract_time_period(query)
        if not start_ts:
            return "无法从您的问题中解析出明确的时间范围。"

        start_dt_str = datetime.fromtimestamp(start_ts).strftime('%Y-%m-%d %H:%M')
        end_dt_str = datetime.fromtimestamp(end_ts).strftime('%Y-%m-%d %H:%M')
        streamer(f"**解析时间范围**: `{start_dt_str}` 到 `{end_dt_str}`\n")
        
        # 2. 从数据库获取事件
        events = self.memory.get_events_for_period(start_ts, end_ts)
        
        if not events:
            return f"在 {start_dt_str} 到 {end_dt_str} 期间没有发现任何记忆记录。"
            
        context_str = f"以下是从 {start_dt_str} 到 {end_dt_str} 期间，按时间顺序记录的所有活动摘要：\n\n"
        for event in events:
            event_time = datetime.fromtimestamp(event['start_time']).strftime('%Y-%m-%d %H:%M')
            context_str += f"- [{event_time}] {event['summary']}\n"
            
        # 3. 返回上下文给主流程进行最终总结 (也可以在这里直接调用LLM总结)
        # 这里选择返回拼接好的上下文，让主流程的最终生成步骤来完成总结，保持逻辑统一
        return context_str

    def _extract_time_period(self, query: str):
        now = datetime.now()
        prompt = f"""
你是一个时间解析专家。根据用户的问题和当前时间，提取一个开始时间戳和一个结束时间戳。
当前时间: {now.isoformat()}

用户问题: "{query}"

请以JSON格式返回: {{"start_iso": "YYYY-MM-DDTHH:MM:SS", "end_iso": "YYYY-MM-DDTHH:MM:SS"}}
"""
        try:
            response = self.llm_client.chat.completions.create(
                model=config.LLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            time_data = json.loads(response.choices[0].message.content)
            start_dt = datetime.fromisoformat(time_data['start_iso'])
            end_dt = datetime.fromisoformat(time_data['end_iso'])
            return start_dt.timestamp(), end_dt.timestamp()
        except Exception as e:
            logger.error(f"Time extraction failed: {e}", exc_info=True)
            # 提供一个默认的回退，比如“今天”
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return today_start.timestamp(), now.timestamp()