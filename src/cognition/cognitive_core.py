# src/cognition/cognitive_core.py
import base64
import logging
from openai import OpenAI
import config
import json
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class CognitiveCore:
    def __init__(self):
        print(f"  [Cognition] 初始化 LVM Client...")
        if not config.LVM_API_KEY:
            logger.error("❌ LVM_API_KEY 未设置")
        self.lvm_client = OpenAI(api_key=config.LVM_API_KEY, base_url=config.LVM_BASE_URL)
        
        print(f"  [Cognition] 初始化 LLM Client...")
        if not config.LLM_API_KEY:
            logger.error("❌ LLM_API_KEY 未设置")
        self.llm_client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
        
    def analyze_event(self, event_data):
        event_id = event_data['event_id']
        
        # 1. 视觉分析 (传入更多上下文)
        analysis_result = self._visual_analysis_json(event_data)
        
        if not analysis_result:
            print("  ❌ [Cognition] 视觉分析为空")
            return None
            
        summary = analysis_result.get('summary', '无有效描述')
        kg_data = self._extract_kg(summary)
        
        return {
            "summary": summary,
            "kg_data": kg_data,
            "scene_label": analysis_result.get('scene_label', '日常'),
            "interaction_score": analysis_result.get('interaction_score', 0)
        }

    def _visual_analysis_json(self, event_data):
        frames = event_data.get('frames', [])
        if not frames: return None
        
        start_dt = datetime.fromtimestamp(event_data['start_time']).strftime('%H:%M:%S')
        end_dt = datetime.fromtimestamp(event_data['end_time']).strftime('%H:%M:%S')
        
        # --- 关键修复：统计检测到的人数最大值 ---
        max_person_count = 0
        known_names = set()
        
        for f in frames:
            # 统计这一帧有多少个框
            current_count = len(f.get('detections', []))
            if current_count > max_person_count:
                max_person_count = current_count
                
            for d in f.get('detections', []):
                name = d.get('name', 'Unknown')
                if name not in ['Unknown', 'Unknown_Body']:
                    known_names.add(name)
        
        names_str = ", ".join(known_names) if known_names else "无已知身份人员"
        
        # --- Prompt 强逻辑注入 ---
        prompt_text = f"""
        你是一个专业的家庭安防AI助手。请分析监控视频抽帧图片。
        
        【场景元数据】
        - 时间范围: {start_dt} 至 {end_dt}
        - 视觉检测到的**最大同时在场人数**: 【{max_person_count}人】
        - 已识别具体身份: 【{names_str}】 
        
        【任务要求】
        请结合图片内容和元数据，严格以 JSON 格式输出：
        1. "summary": 生成一段连贯的中文描述。
           - 必须明确指出画面中有几个人。
           - 如果有交互（交谈、传递物品、共处），请重点描述。
        2. "scene_label": 从以下标签中选一：[无人闲置, 单人独处, 多人社交, 家庭聚会, 护理服务, 跌倒风险, 异常入侵]。
           - **重要规则**：如果元数据中人数 >= 2，且人们在同一空间，请优先选择 [多人社交] 或 [家庭聚会]，**严禁**选择 [单人独处]。
        3. "interaction_score": 0-10 整数。
           - 单人活动一般 1-3 分。
           - 多人互动一般 4-8 分。
           - 紧急情况 9-10 分。

        【JSON 示例】
        {{
            "summary": "画面中出现两人。张三在沙发上，李四递给他一杯水...",
            "scene_label": "多人社交",
            "interaction_score": 5
        }}
        """
        
        # 采样逻辑 (均匀采样15张)
        MAX_IMAGES = 15 
        total_frames = len(frames)
        if total_frames <= MAX_IMAGES:
            indices = range(total_frames)
        else:
            step = (total_frames - 1) / (MAX_IMAGES - 1)
            indices = [int(i * step) for i in range(MAX_IMAGES)]
            
        content = [{"type": "text", "text": prompt_text}]
        
        valid_images = 0
        for idx in indices:
            f = frames[idx]
            try:
                with open(f['image_path'], "rb") as img:
                    b64 = base64.b64encode(img.read()).decode()
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                    valid_images += 1
            except: pass

        if valid_images == 0: return None

        try:
            resp = self.lvm_client.chat.completions.create(
                model=config.LVM_MODEL_NAME, 
                messages=[{"role": "user", "content": content}],
                temperature=0.2,
                max_tokens=1000, 
                response_format={"type": "json_object"}
            )
            return self._clean_and_parse_json(resp.choices[0].message.content)
        except Exception as e:
            logger.error(f"视觉分析失败: {e}")
            return None

    def _extract_kg(self, text):
        prompt = f"提取实体和关系(JSON): {text}"
        try:
            resp = self.llm_client.chat.completions.create(
                model=config.LLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            return self._clean_and_parse_json(resp.choices[0].message.content)
        except: return {"entities": [], "relationships": []}

    def _clean_and_parse_json(self, raw_text):
        try: return json.loads(raw_text)
        except:
            text = raw_text.replace("```json", "").replace("```", "").strip()
            match = re.search(r'\{.*\}', text, re.DOTALL)
            return json.loads(match.group()) if match else {}
