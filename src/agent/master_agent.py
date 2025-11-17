# src/agent/master_agent.py (真正完整的最终版)
import logging
from openai import OpenAI
import config
from src.memory.long_term_memory import LongTermMemory
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class MasterAgent:
    def __init__(self, memory: LongTermMemory):
        self.memory = memory
        self.llm_client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
        logger.info("MasterAgent initialized.")

    # --- 专家方法 (私有) ---

    def _get_query_route(self, query: str) -> str:
        prompt = f"""
你是一个任务分析与路由专家。根据用户的问题，判断最适合处理该问题的专家。
可用专家如下：
- `memory_retrieval`: 当问题是关于具体的、近期发生的事件，需要回忆某个场景或动作时。例如："我刚才把钥匙放哪了？", "lizhijun下午在做什么？"
- `graph_reasoning`: 当问题是关于实体之间的关系、规律、频率或统计时，需要进行逻辑推理。例如："我和谁一起出现过？", "我的杯子通常放在哪里？"
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
        valid_routes = ["memory_retrieval", "graph_reasoning", "summarization"]
        return route if route in valid_routes else "memory_retrieval"

    def _refine_query_for_retrieval(self, query: str) -> str:
        prompt = f"""
你是一个查询优化专家。用户的原始问题可能很口语化。请将其改写成一个更适合向量数据库进行语义搜索的陈述句或关键词组。

原始问题: "{query}"
优化后的查询:
"""
        response = self.llm_client.chat.completions.create(
            model=config.LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()

    def _extract_time_period(self, query: str):
        now = datetime.now()
        prompt = f"""
你是一个时间解析专家。根据用户的问题和当前时间，提取一个开始时间和结束时间。
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
        except Exception:
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return today_start.timestamp(), now.timestamp()
            
    def _memory_retrieval_expert(self, query: str, streamer):
        streamer("**调用专家**: `记忆检索专家`\n")
        yield "**决策路径**: `记忆检索`\n正在优化您的问题... 🧐"
        
        refined_query = self._refine_query_for_retrieval(query)
        streamer(f"**优化查询**: `{query}` -> `{refined_query}`\n")
        yield f"**决策路径**: `记忆检索`\n**优化查询**: `{refined_query}`\n正在进行向量搜索... 🧠"

        retrieved_events = self.memory.semantic_search(refined_query, top_k=4)
        
        if not retrieved_events:
            return None, None
        
        context_str = "以下是我检索到的相关记忆片段：\n\n"
        for i, event in enumerate(retrieved_events):
            time_str = datetime.fromtimestamp(event['start_time']).strftime('%Y-%m-%d %H:%M:%S')
            context_str += f"--- 记忆片段 {i+1} [{time_str}] ---\n摘要: {event['summary']}\n\n"
            
        return context_str, retrieved_events

    def _graph_reasoning_expert(self, query: str, streamer):
        streamer("**调用专家**: `知识图谱推理专家`\n")
        yield "**决策路径**: `知识图谱推理`\n正在将您的问题转换为数据库查询... ⚙️"
        
        result = self.memory.query_knowledge_graph_by_nl(query)
        streamer(f"**知识图谱查询结果**: \n{result}\n")
        
        context_str = f"以下是我从知识图谱中查询到的信息：\n\n{result}"
        return context_str, None # KG查询没有直接的“事件”证据

    def _summarization_expert(self, query: str, streamer):
        streamer("**调用专家**: `活动总结专家`\n")
        yield "**决策路径**: `活动总结`\n正在解析您问题中的时间范围... 📅"
        
        start_ts, end_ts = self._extract_time_period(query)
        start_dt_str = datetime.fromtimestamp(start_ts).strftime('%Y-%m-%d %H:%M')
        end_dt_str = datetime.fromtimestamp(end_ts).strftime('%Y-%m-%d %H:%M')
        streamer(f"**解析时间范围**: `{start_dt_str}` 到 `{end_dt_str}`\n")
        yield f"**决策路径**: `活动总结`\n**时间范围**: `{start_dt_str}` -> `{end_dt_str}`\n正在从数据库筛选事件... 📚"
        
        events = self.memory.get_events_for_period(start_ts, end_ts)
        
        if not events:
            return f"在 {start_dt_str} 到 {end_dt_str} 期间没有发现任何记忆记录。", None
            
        context_str = f"以下是从 {start_dt_str} 到 {end_dt_str} 期间，按时间顺序记录的所有活动摘要：\n\n"
        for event in events:
            event_time = datetime.fromtimestamp(event['start_time']).strftime('%Y-%m-%d %H:%M')
            context_str += f"- [{event_time}] {event['summary']}\n"
            
        return context_str, None # 总结任务的“证据”就是上下文本身，不单独展示

    def _generate_final_answer(self, query, context):
        """生成最终答案的生成器"""
        system_prompt = "你是一个AI Agent的记忆核心。请基于以下提供的【背景上下文】，用自然、流畅的口吻回答用户的问题。如果信息不足，就坦诚说明。"
        user_prompt = f"【背景上下文】\n{context}\n\n【用户问题】\n{query}"
        
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = self.llm_client.chat.completions.create(model=config.LLM_MODEL_NAME, messages=messages, stream=True, temperature=0.5)
        
        for chunk in response:
            if chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # --- 主执行流程 (公共) ---
    def execute_query_steps(self, query, streamer):
        """分步执行查询并yield中间状态"""
        # 1. 路由
        streamer("正在分析问题...\n")
        yield {"status": "routing", "content": "正在分析您的问题... 🤔"}
        route = self._get_query_route(query)
        streamer(f"决策路径: {route}\n")
        yield {"status": "routing", "content": f"**决策路径**: `{route}`"}

        # 2. 获取上下文 (BINGO! 已为所有分支实现)
        context, evidence = None, None
        context_generator = None

        if route == "memory_retrieval":
            context_generator = self._memory_retrieval_expert(query, streamer)
        elif route == "graph_reasoning":
            context_generator = self._graph_reasoning_expert(query, streamer)
        elif route == "summarization":
            context_generator = self._summarization_expert(query, streamer)
        
        # 消费生成器并yield中间状态
        if context_generator:
            # 检查它是不是一个生成器
            if hasattr(context_generator, '__iter__') and not isinstance(context_generator, (str, tuple)):
                for step_output in context_generator:
                    if isinstance(step_output, str): # 中间状态更新
                        yield {"status": "retrieving", "content": step_output}
                    else: # 最终结果 (context, evidence)
                        context, evidence = step_output
            else: # 如果不是生成器，直接获取结果
                context, evidence = context_generator

        if not context:
            yield {"status": "done", "content": "抱歉，我没有找到与您问题相关的信息。"}
            return

        yield {"status": "generating", "content": f"**决策路径**: `{route}`\n信息检索完毕，正在生成回答... ✍️"}

        # 3. 生成最终答案
        final_answer = ""
        answer_generator = self._generate_final_answer(query, context)
        for chunk in answer_generator:
            final_answer += chunk
            yield {"status": "generating", "content": final_answer, "evidence": evidence}

        # 4. 最终完成
        yield {"status": "done", "content": final_answer, "evidence": evidence}