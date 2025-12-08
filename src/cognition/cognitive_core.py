# src/cognition/cognitive_core.py
import base64
import logging
from openai import OpenAI
import config
import json
import re

logger = logging.getLogger(__name__)

class CognitiveCore:
    def __init__(self):
        print(f"  [Cognition] 初始化 AI Client...")
        # 确保使用 config 中正确加载的 Key
        if not config.API_KEY:
            logger.error("❌ API_KEY 未设置，认知模块将无法工作！")
        
        self.client = OpenAI(
            api_key=config.API_KEY, 
            base_url=config.BASE_URL
        )
        
    def analyze_event(self, event_data):
        event_id = event_data['event_id']
        print(f"  🧠 [Cognition] 开始分析事件 {event_id}...")
        
        # 1. 视觉分析
        analysis_result = self._visual_analysis_json(event_data)
        if not analysis_result:
            print("  ❌ [Cognition] 视觉分析失败或为空")
            return None
            
        summary = analysis_result.get('summary', '无有效描述')
        
        # 2. 知识图谱提取 (增加重试和清洗逻辑)
        kg_data = self._extract_kg(summary)
        
        print(f"  ✅ [Cognition] 完成: {analysis_result.get('scene_label')} | KG实体数: {len(kg_data.get('entities', []))}")
        
        return {
            "summary": summary,
            "kg_data": kg_data,
            "scene_label": analysis_result.get('scene_label', '日常'),
            "interaction_score": analysis_result.get('interaction_score', 0)
        }

    def _visual_analysis_json(self, event_data):
        frames = event_data.get('frames', [])
        if not frames: return None
        
        # 构建 Prompt
        prompt_text = """
        你是一个智能监控分析员。请分析图片序列，严格输出 JSON 格式：
        {
            "summary": "详细描述画面中人物的行为、动作、神态以及与环境的交互。",
            "scene_label": "从[无人/单人独处/多人社交/护理/跌倒风险]中选一个",
            "interaction_score": 0-10的整数(10为最紧急)
        }
        注意：直接返回 JSON 字符串，不要Markdown代码块。
        """
        
        content = [{"type": "text", "text": prompt_text}]
        
        # 限制图片数量，防止Token溢出，取首中尾
        selected_frames = [frames[0], frames[len(frames)//2], frames[-1]] if len(frames) >= 3 else frames
        
        for f in selected_frames:
            try:
                with open(f['image_path'], "rb") as img:
                    b64 = base64.b64encode(img.read()).decode()
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            except Exception as e:
                logger.warning(f"图片读取失败: {e}")

        try:
            resp = self.client.chat.completions.create(
                model=config.AI_VL_MODEL,
                messages=[{"role": "user", "content": content}],
                temperature=0.2,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            return self._clean_and_parse_json(resp.choices[0].message.content)
        except Exception as e:
            logger.error(f"视觉API调用错误: {e}")
            return None

    def _extract_kg(self, text):
        """从摘要中提取知识图谱"""
        prompt = f"""
        基于文本提取实体和关系。
        文本: "{text}"
        
        返回格式(JSON):
        {{
            "entities": [{{"name": "张三", "type": "Person"}}, {{"name": "沙发", "type": "Object"}}],
            "relationships": [{{"source": "张三", "target": "沙发", "type": "坐"}}]
        }}
        """
        try:
            resp = self.client.chat.completions.create(
                model=config.AI_THINKING_MODEL, # 使用纯文本模型即可，更便宜快
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            return self._clean_and_parse_json(resp.choices[0].message.content)
        except Exception as e:
            logger.warning(f"KG提取失败: {e}")
            return {"entities": [], "relationships": []}

    def _clean_and_parse_json(self, raw_text):
        """增强的 JSON 清洗解析器"""
        try:
            # 1. 尝试直接解析
            return json.loads(raw_text)
        except:
            # 2. 去除 Markdown 代码块
            text = raw_text.replace("```json", "").replace("```", "").strip()
            # 3. 尝试提取 {} 之间的内容
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
            logger.error(f"无法解析JSON: {raw_text[:50]}...")
            return {}