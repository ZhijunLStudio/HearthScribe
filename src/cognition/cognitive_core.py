# src/cognition/cognitive_core.py
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
        with open(image_path, "rb") as img_file: return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        return None

class CognitiveCore:
    def __init__(self):
        self.lvm_client = OpenAI(api_key=config.LVM_API_KEY, base_url=config.LVM_BASE_URL)
        self.llm_client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
        logger.info("Cognitive Core initialized.")

    def analyze_event(self, event_data):
        event_id = event_data['event_id']
        logger.info(f"--- Starting cognitive analysis for event {event_id} ---")
        summary = self._generate_visual_summary(event_data)
        if not summary:
            logger.error(f"Event {event_id}: Failed to generate summary. Aborting.")
            return None
        kg_data = self._extract_knowledge_graph(summary)
        if not kg_data:
            logger.warning(f"Event {event_id}: KG extraction failed, but summary is saved.")
            kg_data = {"entities": [], "relationships": []}
        return {"summary": summary, "kg_data": kg_data}

    def _generate_visual_summary(self, event_data):
        """
        调用 LVM 生成摘要。
        BINGO! 现在会动态构建Prompt，将已识别的人名告诉LVM。
        """
        frames = event_data.get("frames", [])
        if not frames: return None
        
        # --- 1. 提取上下文信息：识别到的人名 ---
        identities_seen = set()
        for frame_data in frames:
            for detection in frame_data.get('detections', []):
                name = detection.get('name')
                if name and name != 'Unknown':
                    identities_seen.add(name)
        
        known_identities_list = list(identities_seen)
        
        # --- 2. 动态构建Prompt ---
        indices = np.linspace(0, len(frames) - 1, min(len(frames), 5), dtype=int)
        
        prompt_intro = "你是一个智能家庭助手。请分析这些按时间顺序排列的图片，它们描述了一个连贯的事件。"
        
        # 如果识别到了人名，就明确告诉LVM
        if known_identities_list:
            context_prompt = f"场景中已知的人物有: {', '.join(known_identities_list)}。在你的描述中，请务必使用这些名字来指代他们。"
        else:
            context_prompt = "场景中没有识别出已知人物。"

        prompt_task = """
请生成一段简洁、准确的中文摘要，描述发生了什么。重点关注：
1. 谁在画面中？(如果名字在已知人物列表中，请使用他们的名字)
2. 他们在做什么具体的动作？与什么物体进行了交互？
3. 事件发生的地点或场景是什么？
摘要应直接描述事实，不要任何开场白或客套话。
"""
        # 组合成最终的prompt
        final_prompt = f"{prompt_intro}\n{context_prompt}\n{prompt_task}"
        
        logger.info(f"Generated LVM prompt with context: {context_prompt}")

        # --- 3. 准备API请求 (与之前相同) ---
        messages = [{"role": "user", "content": [{"type": "text", "text": final_prompt}]}]
        for idx in indices:
            b64 = image_to_base64(frames[idx]["image_path"])
            if b64: messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        
        if len(messages[0]["content"]) < 2:
            logger.error("No valid images to send to LVM.")
            return None

        # --- 4. 调用API (与之前相同) ---
        try:
            t0 = time.time()
            resp = self.lvm_client.chat.completions.create(model=config.LVM_MODEL_NAME, messages=messages, max_tokens=500, temperature=0.3)
            summary = resp.choices[0].message.content.strip()
            logger.info(f"LVM Summary generated in {time.time()-t0:.2f}s: {summary}")
            return summary
        except Exception as e:
            logger.error(f"LVM call failed: {e}", exc_info=True)
            return None

    # src/cognition/cognitive_core.py (修改后的新代码)

    def _extract_knowledge_graph(self, summary_text):
        prompt = f"""
从以下摘要中提取知识图谱。
实体类型只能是: ["Person", "Object", "Location", "Activity"]
关系类型尽量简短 (如 "picked_up", "is_in")

【输入摘要】: "{summary_text}"

【输出格式】: 必须是合法的 JSON，并用 markdown 代码块包裹: ```json\n{{"entities": [...], "relationships": [...]}}\n```
"""
        try:
            t0 = time.time()
            resp = self.llm_client.chat.completions.create(
                model=config.LLM_MODEL_NAME, 
                messages=[{"role": "user", "content": prompt}], 
                temperature=0.1,
                # BINGO! 很多模型支持JSON模式，直接让它输出合法的JSON
                response_format={"type": "json_object"} 
            )
            raw_response = resp.choices[0].message.content.strip()
            
            # --- 解析JSON ---
            # 现代OpenAI兼容的API支持 `response_format`,可以直接解析
            try:
                kg_data = json.loads(raw_response)
                logger.info(f"KG Extracted in {time.time()-t0:.2f}s using direct JSON mode.")
                return kg_data
            except json.JSONDecodeError:
                logger.warning("Direct JSON parsing failed. Falling back to regex extraction.")
                # 如果直接解析失败（可能API不支持或返回了额外文本），则使用下面的后备方法
                pass

            # --- 后备方法：正则表达式提取 ---
            # 1. 优先匹配 ```json ... ``` 代码块
            match = re.search(r'```json\s*(\{.*?\})\s*```', raw_response, re.DOTALL)
            if not match:
                # 2. 如果没有代码块，再尝试匹配裸露的 {...}
                match = re.search(r'(\{.*?\})', raw_response, re.DOTALL)

            if match:
                json_str = match.group(1)
                try:
                    kg_data = json.loads(json_str)
                    logger.info(f"KG Extracted in {time.time()-t0:.2f}s using regex fallback.")
                    return kg_data
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode extracted JSON string: {json_str}", exc_info=True)
                    logger.error(f"JSONDecodeError: {e}")
                    return None
            
            logger.error(f"Failed to find any JSON in LLM response: {raw_response}")
            return None

        except Exception as e:
            logger.error(f"LLM KG extraction failed with an unexpected error: {e}", exc_info=True)
            return None