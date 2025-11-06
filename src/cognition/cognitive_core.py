# src/cognition/cognitive_core.py (完整新代码)
import base64
import json
from openai import OpenAI
import numpy as np
import config
import logging
import time
import re

logger = logging.getLogger(__name__)

def image_to_base_64(image_path):
    try:
        with open(image_path, "rb") as img_file: 
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error converting image to base64 for path {image_path}: {e}")
        return None


class CognitiveCore:
    def __init__(self):
        self.lvm_client = OpenAI(api_key=config.LVM_API_KEY, base_url=config.LVM_BASE_URL)
        self.llm_client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
        logger.info("Cognitive Core initialized.")

    def analyze_event(self, event_data):
        event_id = event_data['event_id']
        logger.info(f"--- Starting cognitive analysis for event {event_id} ---")
        
        initial_summary = self._generate_visual_summary(event_data)
        if not initial_summary:
            logger.error(f"Event {event_id}: Failed to generate initial summary. Aborting.")
            return None
            
        final_summary = self._critique_and_refine_summary(initial_summary, event_data)
        
        kg_data = self._extract_knowledge_graph(final_summary)
        if not kg_data:
            logger.warning(f"Event {event_id}: KG extraction failed, but summary is saved.")
            kg_data = {"entities": [], "relationships": []}
            
        return {"summary": final_summary, "kg_data": kg_data}

    def _critique_and_refine_summary(self, summary, event_data):
        logger.info(f"Critiquing initial summary: {summary}")
        
        critique_prompt = f"""
你是一个严谨的细节审查官。这里有一段由AI生成的事件摘要，请对其进行评估。

【初步摘要】: "{summary}"

【评估标准】:
1.  **准确性**: 摘要是否准确描述了图片中的核心动作和交互？
2.  **完整性**: 是否遗漏了任何关键人物、物体或明显的意图？
3.  **客观性**: 摘要是否只是描述事实，没有过多主观臆断？

【你的任务】:
请基于以上标准，给出一个简短的批判性评估。如果摘要质量很高，请回答 "摘要质量高，无需修改。"。如果存在问题，请以 "需要优化，建议如下：" 开头，并给出具体的修改建议。
"""
        try:
            messages = self._prepare_lvm_messages("placeholder", event_data)
            if len(messages[0]['content']) < 2:
                logger.warning("No valid images for critique, skipping critique step.")
                return summary

            messages[0]["content"][0]['text'] = critique_prompt

            resp = self.lvm_client.chat.completions.create(model=config.LVM_MODEL_NAME, messages=messages, max_tokens=300, temperature=0.2)
            critique = resp.choices[0].message.content.strip()
            logger.info(f"Critique received: {critique}")

            if "无需修改" in critique:
                logger.info("Summary approved by critic.")
                return summary
            
            logger.info("Refining summary based on critique...")
            refine_prompt = f"""
你是一个优秀的事件描述作者。请根据一位审查官的修改建议，重写并优化以下初步摘要。

【初步摘要】: "{summary}"
【审查官建议】: "{critique}"

【你的任务】:
生成一段最终的、更完善的摘要。请直接输出最终摘要，不要包含任何解释性文字。
"""
            resp = self.llm_client.chat.completions.create(model=config.LLM_MODEL_NAME, messages=[{"role": "user", "content": refine_prompt}], max_tokens=500, temperature=0.4)
            refined_summary = resp.choices[0].message.content.strip()
            logger.info(f"Refined summary: {refined_summary}")
            return refined_summary

        except Exception as e:
            logger.error(f"Failed to critique/refine summary: {e}", exc_info=True)
            return summary

    def _prepare_lvm_messages(self, final_prompt, event_data):
        frames = event_data.get("frames", [])
        indices = np.linspace(0, len(frames) - 1, min(len(frames), 5), dtype=int)
        
        messages = [{"role": "user", "content": [{"type": "text", "text": final_prompt}]}]
        for idx in indices:
            b64 = image_to_base_64(frames[idx]["image_path"])
            if b64: messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        
        return messages

    def _generate_visual_summary(self, event_data):
        frames = event_data.get("frames", [])
        if not frames: return None
        
        identities_seen = set()
        for frame_data in frames:
            for detection in frame_data.get('detections', []):
                name = detection.get('name')
                if name and name != 'Unknown':
                    identities_seen.add(name)
        
        known_identities_list = list(identities_seen)
        
        prompt_intro = "你是一个智能家庭助手。请分析这些按时间顺序排列的图片，它们描述了一个连贯的事件。"
        
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
        final_prompt = f"{prompt_intro}\n{context_prompt}\n{prompt_task}"
        logger.info(f"Generated LVM prompt with context: {context_prompt}")
        
        messages = self._prepare_lvm_messages(final_prompt, event_data)
        if len(messages[0]["content"]) < 2:
            logger.error("No valid images to send to LVM.")
            return None

        try:
            t0 = time.time()
            resp = self.lvm_client.chat.completions.create(model=config.LVM_MODEL_NAME, messages=messages, max_tokens=500, temperature=0.3)
            summary = resp.choices[0].message.content.strip()
            logger.info(f"LVM Summary generated in {time.time()-t0:.2f}s: {summary}")
            return summary
        except Exception as e:
            logger.error(f"LVM call failed: {e}", exc_info=True)
            return None

    def _extract_knowledge_graph(self, summary_text):
        # BINGO! --- 这是本次修改的核心 ---
        # 我们需要将所有用于示例的 `{}` 替换为 `{{}}` 进行转义
        prompt = f"""
从以下摘要中提取知识图谱。
实体类型只能是: ["Person", "Object", "Location", "Activity"]
关系类型尽量简短 (如 "picked_up", "is_in")

【输入摘要】: "{summary_text}"

【输出格式】: 必须是严格合法的JSON对象。JSON必须包含 'entities' 和 'relationships' 两个键。如果找不到任何实体或关系，则返回 `{{"entities": [], "relationships": []}}`。
请将JSON对象包裹在 markdown 代码块中，像这样: ```json\n{{...}}\n```
"""
        try:
            t0 = time.time()
            resp = self.llm_client.chat.completions.create(
                model=config.LLM_MODEL_NAME, 
                messages=[{"role": "user", "content": prompt}], 
                temperature=0.1,
                response_format={"type": "json_object"} 
            )
            raw_response = resp.choices[0].message.content.strip()
            logger.info(f"LLM raw response for KG: {raw_response}")

            json_str = None
            match = re.search(r'```json\s*(\{.*?\})\s*```', raw_response, re.DOTALL)
            if match:
                json_str = match.group(1)
                logger.info("Found JSON inside markdown code block.")
            else:
                match = re.search(r'(\{.*?\})', raw_response, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    logger.info("Found JSON using non-greedy regex.")

            if json_str:
                try:
                    json_str = json_str.strip().replace('\\n', '\n')
                    json_str = re.sub(r',\s*([\}\]])', r'\1', json_str)
                    
                    kg_data = json.loads(json_str)
                    logger.info(f"KG Extracted in {time.time()-t0:.2f}s. Parsed Data: {json.dumps(kg_data, ensure_ascii=False)}")
                    return kg_data
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode extracted JSON string: {json_str}", exc_info=True)
                    logger.error(f"JSONDecodeError: {e}")
                    return None
            
            logger.error(f"Failed to find any valid JSON in LLM response.")
            return None

        except Exception as e:
            # 捕获 ValueError: Invalid format specifier
            logger.error(f"LLM KG extraction failed. Potentially an f-string issue in the prompt or another error: {e}", exc_info=True)
            return None