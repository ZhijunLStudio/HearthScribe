import os
from pathlib import Path
from paddlex import create_pipeline
import logging

logging.basicConfig(level=logging.INFO)

# --- 配置 ---
GALLERY_ROOT = "known_faces"    # 照片存放目录: known_faces/奶奶/1.jpg
INDEX_SAVE_DIR = "face_index"   # 索引保存路径

def build_face_index():
    print(f"🚀 初始化 PaddleX 人脸识别产线...")
    # 自动下载并加载模型
    pipeline = create_pipeline(pipeline="face_recognition")
    
    gallery_imgs, gallery_labels = [], []
    root = Path(GALLERY_ROOT)
    
    if not root.exists():
        print(f"⚠️ 目录 {GALLERY_ROOT} 不存在，请先创建并放入照片。")
        return

    print(f"📂 正在扫描底库...")
    for person_dir in root.iterdir():
        if person_dir.is_dir():
            for img_path in person_dir.glob("*.*"):
                 if img_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                    gallery_imgs.append(str(img_path))
                    gallery_labels.append(person_dir.name) # 文件夹名即为人名
    
    if not gallery_imgs:
        print("❌ 未找到图片。")
        return

    print(f"📸 开始构建索引，共 {len(gallery_imgs)} 张图片 (可能需要几分钟)...")
    try:
        # 构建 IVF 索引
        index_data = pipeline.build_index(
            gallery_imgs=gallery_imgs, 
            gallery_label=gallery_labels, 
            index_type="IVF", 
            metric_type="IP"
        )
        index_data.save(INDEX_SAVE_DIR)
        print(f"✅ 索引构建成功！已保存至 {INDEX_SAVE_DIR}")
    except Exception as e:
        print(f"❌ 索引构建失败: {e}")

if __name__ == "__main__":
    build_face_index()