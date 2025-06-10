import torch
from ultralytics import YOLO

# 加载 YOLOv8n 模型
model = YOLO('weights/mushroom3.0_v8n.pt')

# 将模型设置为评估模式
model.model.eval()

# 定义一个示例输入（假设输入图像大小为 640x640）
dummy_input = torch.randn(1, 3, 640, 640)

# 进行动态量化
quantized_model = torch.quantization.quantize_dynamic(
    model.model,  # 模型
    {torch.nn.Linear},  # 要量化的层
    dtype=torch.qint8  # 量化类型
)

# 保存量化后的模型
torch.save(quantized_model.state_dict(), 'weights/mushroom3.0_v8n_quantized.pt')
print("模型量化完成，已保存为 'weights/mushroom3.0_v8n_quantized.pt'")