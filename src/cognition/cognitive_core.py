import base64
import logging
from openai import OpenAI
import config
import json

logger = logging.getLogger(__name__)

class CognitiveCore:
    def __init__(self):
        self.client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)
        
    def analyze_event(self, event_data):
        """分析事件：生成摘要 + 提取健康图谱"""
        summary = self._generate_care_summary(event_data)
        if not summary: return None
        
        kg_data = self._extract_care_kg(summary)
        return {"summary": summary, "kg_data": kg_data}

    def _generate_care_summary(self, event_data):
        frames = event_data.get('frames', [])
        if not frames: return None
        
        # 提取识别到的人名
        names = set()
        for f in frames:
            for d in f['detections']:
                if d['name'] not in ['Unknown', 'Unknown_Body']:
                    names.add(d['name'])
        people_str = ", ".join(names) if names else "一位长者"

        # --- 看护专用 Prompt ---
        prompt_content = [
            {"type": "text", "text": f"""
            你是一个专业的家庭养老看护助手。画面中主要人物是：{people_str}。
            请分析这一系列图片，生成一份简短的【安全与健康日志】。
            
            请重点关注以下维度：
            1. **行为**：在做什么？（如：看电视、吃饭、吃药、睡觉、做家务）
            2. **姿态与风险**：是否有跌倒、长时间未动、步态不稳等风险？
            3. **情绪**：看起来心情如何？
            4. **环境**：是否有危险物品（如未关的炉灶）？
            
            输出要求：客观、准确，直接描述事实。
            """}
        ]
        
        # 采样关键帧 (最多5张)
        step = max(1, len(frames) // 5)
        for f in frames[::step][:5]:
            with open(f['image_path'], "rb") as img:
                b64 = base64.b64encode(img.read()).decode()
                prompt_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                
        try:
            resp = self.client.chat.completions.create(
                model=config.AI_VL_MODEL,
                messages=[{"role": "user", "content": prompt_content}],
                temperature=0.3,
                max_tokens=500
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"视觉分析失败: {e}")
            return None

    def _extract_care_kg(self, text):
        """提取结构化健康数据"""
        prompt = f"""
        从以下看护日志中提取关键信息，返回JSON格式。
        日志：{text}
        
        需要提取的字段：
        - entities: list of {{"name": "实体名", "type": "Person/Object/Location/Activity"}}
        - risks: list of string (识别到的潜在风险，如"跌倒风险", "忘记吃药"，无风险则为空)
        - mood: string (识别到的情绪，如"平静", "开心", "焦虑")
        
        只返回JSON字符串，不要Markdown。
        """
        try:
            resp = self.client.chat.completions.create(
                model=config.AI_THINKING_MODEL, # 使用思考模型处理逻辑
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = resp.choices[0].message.content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            logger.error(f"知识抽取失败: {e}")
            return {"entities": [], "risks": [], "mood": "未知"}