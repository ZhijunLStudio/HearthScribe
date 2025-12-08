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
        print(f"  [Cognition] 初始化 AI Client...")
        # 确保 Key 存在
        if not config.API_KEY:
            logger.error("❌ API_KEY 未设置，认知模块将无法工作！请检查 config.py 或 .env")
        
        self.client = OpenAI(
            api_key=config.API_KEY, 
            base_url=config.BASE_URL
        )
        
    def analyze_event(self, event_data):
        event_id = event_data['event_id']
        # print(f"  🧠 [Cognition] 开始分析事件 {event_id}...")
        
        # 1. 视觉分析 (核心：注入时间与人名)
        analysis_result = self._visual_analysis_json(event_data)
        
        if not analysis_result:
            print("  ❌ [Cognition] 视觉分析返回为空，跳过此事件。")
            return None
            
        summary = analysis_result.get('summary', '无有效描述')
        
        # 2. 知识抽取
        kg_data = self._extract_kg(summary)
        
        # 打印简单日志
        # print(f"  ✅ [Cognition] 场景: {analysis_result.get('scene_label')} | 评分: {analysis_result.get('interaction_score')}")
        
        return {
            "summary": summary,
            "kg_data": kg_data,
            "scene_label": analysis_result.get('scene_label', '日常'),
            "interaction_score": analysis_result.get('interaction_score', 0)
        }

    def _visual_analysis_json(self, event_data):
        frames = event_data.get('frames', [])
        if not frames: return None
        
        # --- 关键修改 1: 提取时间元数据 ---
        # 格式化时间戳，例如 "2023-10-27 14:30:05"
        start_dt = datetime.fromtimestamp(event_data['start_time']).strftime('%Y-%m-%d %H:%M:%S')
        end_dt = datetime.fromtimestamp(event_data['end_time']).strftime('%H:%M:%S')
        
        # --- 关键修改 2: 提取已知人名 ---
        # 遍历所有帧的 detections，收集非 Unknown 的名字
        known_names = set()
        for f in frames:
            for d in f.get('detections', []):
                name = d.get('name', 'Unknown')
                if name not in ['Unknown', 'Unknown_Body']:
                    known_names.add(name)
        
        names_str = ", ".join(known_names) if known_names else "无已知身份人员"
        
        # --- 关键修改 3: 构建包含元数据的 Prompt ---
        prompt_text = f"""
        你是一个专业的家庭安防AI助手。请分析提供的监控视频关键帧（图片已包含检测框和人名标注）。
        
        【场景元数据】
        - 时间范围: {start_dt} 至 {end_dt}
        - 已识别人物: 【{names_str}】 
          (注意：如果图片上的检测框标注了名字，请务必在描述中使用该名字；如果标注为Unknown，则描述为陌生人)
        
        【任务要求】
        请严格以 JSON 格式输出分析结果，包含以下字段：
        1. "summary": 生成一段连贯的中文描述。必须包含：
           - 具体时间点（或时间段）。
           - 具体人物名字（谁）。
           - 具体的动作、交互和环境细节（做了什么）。
        2. "scene_label": 从以下标签中选择最贴切的一个：[无人闲置, 单人独处, 多人社交, 家庭聚会, 护理服务, 跌倒风险, 异常入侵]。
        3. "interaction_score": 给出 0-10 的整数评分 (0为无人，10为紧急事件/极高频互动)。

        【JSON 示例】
        {{
            "summary": "在14:30分左右，张三独自坐在客厅沙发上...",
            "scene_label": "单人独处",
            "interaction_score": 2
        }}
        """
        
        content = [{"type": "text", "text": prompt_text}]
        
        # 图片采样：取首、中、尾 3 张，避免 token 过多
        # MemoryStream 保存的图片通常已经画上了框和名字
        indices = [0, len(frames)//2, -1] if len(frames) >= 3 else range(len(frames))
        
        valid_images = 0
        for idx in indices:
            f = frames[idx]
            try:
                with open(f['image_path'], "rb") as img:
                    b64 = base64.b64encode(img.read()).decode()
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                    valid_images += 1
            except Exception as e:
                logger.warning(f"读取图片失败: {f['image_path']} - {e}")

        if valid_images == 0:
            return None

        try:
            # print(f"      -> 发送 {valid_images} 张图片给大模型...")
            resp = self.client.chat.completions.create(
                model=config.AI_VL_MODEL, # 确保 config 中定义了视觉模型
                messages=[{"role": "user", "content": content}],
                temperature=0.2,
                max_tokens=800,
                response_format={"type": "json_object"}
            )
            return self._clean_and_parse_json(resp.choices[0].message.content)
        except Exception as e:
            logger.error(f"视觉分析 API 调用失败: {e}")
            return None

    def _extract_kg(self, text):
        """从文本中提取知识图谱实体和关系"""
        prompt = f"""
        从以下文本中提取实体(Entities)和关系(Relationships)。
        文本: "{text}"
        
        请严格返回 JSON 格式:
        {{
            "entities": [{{"name": "张三", "type": "Person"}}, {{"name": "沙发", "type": "Object"}}],
            "relationships": [{{"source": "张三", "target": "沙发", "type": "坐在", "relation": "sitting_on"}}]
        }}
        """
        try:
            resp = self.client.chat.completions.create(
                model=config.AI_THINKING_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            return self._clean_and_parse_json(resp.choices[0].message.content)
        except Exception:
            return {"entities": [], "relationships": []}

    def _clean_and_parse_json(self, raw_text):
        """鲁棒的 JSON 解析器"""
        try:
            # 1. 尝试直接解析
            return json.loads(raw_text)
        except:
            # 2. 如果包含 markdown 代码块，尝试去除
            text = raw_text.replace("```json", "").replace("```", "").strip()
            # 3. 尝试正则提取大括号内容
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
            logger.error(f"JSON 解析失败。原始返回: {raw_text[:100]}...")
            return {}