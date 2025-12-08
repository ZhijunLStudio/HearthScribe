import base64
import logging
from openai import OpenAI
import config
import json

logger = logging.getLogger(__name__)

class CognitiveCore:
    def __init__(self):
        print(f"  [Cognition] 初始化 OpenAI Client (Base: {config.BASE_URL})...")
        self.client = OpenAI(
            api_key=config.API_KEY, 
            base_url=config.BASE_URL
        )
        
    def analyze_event(self, event_data):
        print(f"  🧠 [Cognition] 开始深度分析事件 {event_data['event_id']}...")
        
        # 1. 视觉理解 (生成摘要 + 评分 + 标签)
        # 注意：这里调用的是新写的 _visual_analysis_json
        analysis_result = self._visual_analysis_json(event_data)
        
        if not analysis_result:
            print("  ❌ [Cognition] 视觉分析返回为空")
            return None
            
        summary = analysis_result.get('summary', '无有效描述')
        
        # 2. 知识抽取 (实体)
        kg_data = self._extract_kg(summary)
        
        # 3. 组装最终结果返回给 main.py
        print(f"  ✅ [Cognition] 分析成功: 场景[{analysis_result.get('scene_label')}] 评分[{analysis_result.get('interaction_score')}]")
        
        return {
            "summary": summary,
            "kg_data": kg_data,
            "scene_label": analysis_result.get('scene_label', '未知'),
            "interaction_score": analysis_result.get('interaction_score', 0)
        }

    def _visual_analysis_json(self, event_data):
        frames = event_data.get('frames', [])
        if not frames: return None
        
        # 提取人名
        names = set()
        for f in frames:
            for d in f['detections']:
                if d['name'] not in ['Unknown', 'Unknown_Body']:
                    names.add(d['name'])
        people_str = ", ".join(names) if names else "陌生人/未识别"
        person_count = max([len(f['detections']) for f in frames]) if frames else 0

        # --- 核心 Prompt: 强制 JSON 输出 ---
        prompt_text = f"""
        你是一个空间态势感知AI。画面中检测到 {person_count} 人（身份：{people_str}）。
        
        请完成任务并严格返回 JSON 格式：
        1. **summary**: 简明扼要地描述发生了什么（行为、交互、环境）。
        2. **scene_label**: 从以下标签中选一个最贴切的：[无人闲置] [单人独处] [多人社交] [护理服务] [家庭聚会] [异常/风险]
        3. **interaction_score**: 态势评分 (0-10)。
           - 0: 无人。
           - 1-3: 单人活动。
           - 4-6: 多人共处/简单交流。
           - 7-9: 深度交互/密切护理。
           - 10: 紧急事件（跌倒/求救）。

        返回格式示例：
        {{
            "summary": "张三坐在沙发上看电视...",
            "scene_label": "单人独处",
            "interaction_score": 2
        }}
        """
        
        prompt_content = [{"type": "text", "text": prompt_text}]
        
        # 采样 3 张图
        for f in frames[:3]:
            try:
                with open(f['image_path'], "rb") as img:
                    b64 = base64.b64encode(img.read()).decode()
                    prompt_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            except: pass

        try:
            print(f"      -> 请求大模型 ({config.AI_VL_MODEL}) 进行 JSON 分析...")
            resp = self.client.chat.completions.create(
                model=config.AI_VL_MODEL,
                messages=[{"role": "user", "content": prompt_content}],
                temperature=0.2,
                max_tokens=800,
                response_format={"type": "json_object"} # 关键：强制 JSON 模式
            )
            content = resp.choices[0].message.content
            # 清理可能的 markdown 标记
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            logger.error(f"视觉分析失败: {e}")
            return None

    def _extract_kg(self, text):
        try:
            resp = self.client.chat.completions.create(
                model=config.AI_THINKING_MODEL,
                messages=[{"role": "user", "content": f"提取JSON实体(entities, relationships): {text}"}],
                extra_body={"enable_thinking": True}
            )
            content = resp.choices[0].message.content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except: return {}