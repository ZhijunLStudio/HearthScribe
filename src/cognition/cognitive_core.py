import base64
import json
from openai import OpenAI
import numpy as np
import config
import logging
import time

logger = logging.getLogger(__name__)

# ... (image_to_base64 函数保持不变) ...
def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"无法将图片 {image_path} 转换为 Base64: {e}")
        return None

class CognitiveCore:
    def __init__(self):
        self.lvm_client = OpenAI(api_key=config.LVM_API_KEY, base_url=config.LVM_BASE_URL)
        logger.info("认知核心已初始化。")

    def analyze_event_with_lvm(self, event_data):
        """调用视觉大模型分析事件"""
        
        frames = event_data["frames"]
        # num_frames_to_send = min(len(frames), 5)
        num_frames_to_send = len(frames)
        key_frames_indices = np.linspace(0, len(frames) - 1, num_frames_to_send, dtype=int)
        
        identities_seen = set()
        for idx in key_frames_indices:
            for det in frames[idx]['detections']:
                identities_seen.add(det['name'])
        
        prompt_text = f"""
你是一个专业的家庭场景观察员。请仔细分析以下按时间顺序排列的{len(key_frames_indices)}张关键帧图像，它们共同描述了一个持续约{config.EVENT_MAX_DURATION_SECONDS / 60:.1f}分钟的事件。
场景中出现的人物有: {', '.join(list(identities_seen)) or '未知'}。

你的任务是:
1. 生成一段详细、流畅、有人情味的中文段落作为 "summary"，用故事性的口吻描述在这段时间内发生了什么事。请务必具体描述每个已知人物（如果存在）的行为和互动。
2. 你的所有输出必须是**只包含一个格式正确的JSON对象**，结构如下:
    {{
      "summary": "【这里填写你的详细中文描述】"
    }}
"""
        message_content = [{"type": "text", "text": prompt_text}]
        valid_images = 0
        for idx in key_frames_indices:
            b64_image = image_to_base64(frames[idx]["image_path"])
            if b64_image:
                message_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}})
                valid_images += 1
        
        if valid_images == 0:
            logger.error("没有有效的图片可供分析。")
            return None

        try:
            logger.info(f"准备调用LVM: {config.LVM_MODEL_NAME} 分析事件 {event_data['event_id']} (使用 {valid_images} 张关键帧)...")
            
            # BINGO! 记录开始时间
            start_api_call = time.time()
            
            chat_completion = self.lvm_client.chat.completions.create(
                model=config.LVM_MODEL_NAME,
                messages=[{"role": "user", "content": message_content}],
                max_tokens=2048,
                temperature=0.2
            )
            
            # BINGO! 记录API耗时
            logger.info(f"LVM API调用完成，耗时: {time.time() - start_api_call:.2f} 秒。")

            response_text = chat_completion.choices[0].message.content
            
            # BINGO! 打印从API获取的原始文本，这非常重要！
            logger.info(f"从LVM获取的原始响应: {response_text}")
            
            json_str_match = response_text[response_text.find('{'):response_text.rfind('}')+1]
            if not json_str_match:
                logger.error(f"LVM响应中未找到有效的JSON对象。")
                return None
                
            json_response = json.loads(json_str_match)
            return json_response

        except Exception as e:
            # BINGO! 同样，打印完整的错误堆栈
            logger.error(f"调用视觉大模型或解析JSON失败: {e}", exc_info=True)
            return None