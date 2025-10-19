import base64
import json
from openai import OpenAI
import numpy as np
import config
import logging
import time
import re

logger = logging.getLogger(__name__)

def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        return None

class CognitiveCore:
    def __init__(self):
        self.lvm_client = OpenAI(api_key=config.LVM_API_KEY, base_url=config.LVM_BASE_URL)
        # 复用同一个客户端，因为通常 base_url 是一样的，只是模型不同
        # 如果你的 LLM 和 LVM 用不同的服务商，这里需要分别初始化
        self.llm_client = self.lvm_client 
        logger.info("Cognitive Core initialized.")

    def analyze_event(self, event_data):
        """
        两阶段分析：
        1. LVM: 看图 -> 生成自然语言摘要
        2. LLM: 读摘要 -> 提取结构化知识图谱 (JSON)
        """
        event_id = event_data['event_id']
        logger.info(f"--- Starting cognitive analysis for event {event_id} ---")

        # --- 阶段 1: 视觉摘要 (LVM) ---
        summary = self._generate_visual_summary(event_data)
        if not summary:
            logger.error(f"Event {event_id}: Failed to generate summary. Aborting.")
            return None

        # --- 阶段 2: 知识提取 (LLM) ---
        # 即使这一步失败了，我们也应该返回摘要，不应该让整个事件丢失
        kg_data = self._extract_knowledge_graph(summary)
        if not kg_data:
            logger.warning(f"Event {event_id}: Knowledge graph extraction failed, but summary is saved.")
            kg_data = {"entities": [], "relationships": []} # 返回空结构以保底

        return {"summary": summary, "kg_data": kg_data}

    def _generate_visual_summary(self, event_data):
        """调用 LVM 生成摘要"""
        frames = event_data["frames"]
        if not frames: return None
        
        # 选取最多 5 张关键帧
        indices = np.linspace(0, len(frames) - 1, min(len(frames), 5), dtype=int)
        
        prompt = """
你是一个智能家庭助手。请分析这些按时间顺序排列的图片，它们描述了一个连贯的事件。
请生成一段简洁、准确的中文摘要，描述发生了什么。重点关注：
1. 谁在画面中？(如果认识，请使用他们的名字，如 'lizhijun')
2. 他们在做什么动作？与什么物体进行了交互？
3. 事件发生的地点或场景是什么？
摘要直接描述事实即可，不需要任何开场白。
"""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        for idx in indices:
            b64 = image_to_base64(frames[idx]["image_path"])
            if b64:
                messages[0]["content"].append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                )

        try:
            t0 = time.time()
            resp = self.lvm_client.chat.completions.create(
                model=config.LVM_MODEL_NAME,
                messages=messages,
                max_tokens=500,
                temperature=0.3 # 稍微低一点温度，让描述更准确
            )
            summary = resp.choices[0].message.content.strip()
            logger.info(f"LVM Summary generated in {time.time()-t0:.2f}s: {summary}")
            return summary
        except Exception as e:
            logger.error(f"LVM call failed: {e}", exc_info=True)
            return None

    def _extract_knowledge_graph(self, summary_text):
        """调用 LLM 从文本中提取 KG"""
        prompt = f"""
你的任务是从以下事件摘要中提取结构化的知识图谱。
找出核心的“实体”(Entity)和它们之间的“关系”(Relationship)。

【实体要求】
- name: 实体的名称 (如 "lizhijun", "水杯", "厨房")
- type: 实体类型，只能是以下之一: ["Person", "Object", "Location", "Activity"]

【关系要求】
- source: 源实体名称
- target: 目标实体名称
- type: 关系类型，尽量简短明确 (如 "picked_up", "is_in", "talking_to")

【输入摘要】
"{summary_text}"

【输出格式】
必须是合法的 JSON 格式，不要包含 Markdown 标记：
{{
  "entities": [ {{"name": "...", "type": "..."}}, ... ],
  "relationships": [ {{"source": "...", "target": "...", "type": "..."}}, ... ]
}}
"""
        try:
            t0 = time.time()
            resp = self.llm_client.chat.completions.create(
                model=config.LLM_MODEL_NAME, # 这里用文本大模型就行
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, # 极低温度，强制结构化输出
                # response_format={ "type": "json_object" } # 如果你的API支持，务必加上这行！
            )
            raw_json = resp.choices[0].message.content.strip()
            
            # 强力清洗 JSON，防止大模型偶尔返回 ```json 包裹的内容
            json_str = re.search(r'\{.*\}', raw_json, re.DOTALL)
            if json_str:
                cleaned_json = json_str.group(0)
                kg_data = json.loads(cleaned_json)
                logger.info(f"KG Extracted in {time.time()-t0:.2f}s: {len(kg_data.get('entities',[]))} entities, {len(kg_data.get('relationships',[]))} relations.")
                return kg_data
            else:
                logger.error(f"Failed to find JSON in LLM response: {raw_json}")
                return None

        except Exception as e:
            logger.error(f"LLM KG extraction failed: {e}", exc_info=True)
            return None