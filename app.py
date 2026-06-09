# app.py
# Gradio Web 界面

import gradio as gr
import numpy as np
from inference import BreastCancerPredictor
import os

# 初始化预测器（根据你的模型路径修改）
MODEL_PATH = "./final_model.pth"
DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"

print(f"正在加载模型... (设备: {DEVICE})")
predictor = BreastCancerPredictor(MODEL_PATH, device=DEVICE)
print("✅ 模型加载完成！")


def analyze_image(image):
    """处理上传的图片"""
    # 保存临时图片
    temp_path = "temp_upload.png"
    image.save(temp_path)

    # 推理
    result = predictor.predict(temp_path)

    # 生成热力图叠加
    overlay = predictor.overlay_heatmap(result['image'], result['heatmap'])

    # 清理临时文件
    if os.path.exists(temp_path):
        os.remove(temp_path)

    # 输出文本
    text_output = f"""
    ╔══════════════════════════╗
    ║  乳腺癌病理图像分析结果  ║
    ╠══════════════════════════╣
    ║  诊断结果: {result['prediction']:>6}        ║
    ║  置信度:   {result['confidence']:>6.1%}      ║
    ║  恶性概率: {result['malignant_prob']:>6.4f} ║
    ╚══════════════════════════╝
    """

    return text_output, overlay


# 创建 Gradio 界面
with gr.Blocks(title="乳腺癌病理图像智能分析系统") as demo:
    gr.Markdown("""
    # 🏥 乳腺癌病理图像智能分析系统

    **基于 CTransPath-SiMVC 的多视图深度学习模型**

    - 上传病理组织切片图像（支持 40×/100×/200×/400× 任意倍数）
    - 自动判断良恶性并给出置信度
    - 生成 Grad-CAM 热力图，展示模型关注的癌区位置
    """)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="上传病理图像")
            analyze_btn = gr.Button("🔍 开始分析", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(label="分析结果", lines=8)
            output_heatmap = gr.Image(label="Grad-CAM 癌区定位热力图")

    analyze_btn.click(
        fn=analyze_image,
        inputs=input_image,
        outputs=[output_text, output_heatmap]
    )

    gr.Markdown("""
    ---
    ### 使用说明
    1. 上传一张乳腺癌组织病理图像（HE染色切片）
    2. 点击"开始分析"
    3. 系统将输出良恶性判断、置信度和癌区定位热力图

    ### 技术说明
    - 骨干网络：CTransPath (Swin Transformer)
    - 多视图框架：SiMVC (Simple Multi-View Clustering)
    - 弱监督定位：Grad-CAM
    - 数据集：BreakHis (Breast Cancer Histopathological Database)
    """)

if __name__ == "__main__":
    demo.launch(share=True)
