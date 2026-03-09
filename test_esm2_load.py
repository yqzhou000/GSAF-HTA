import torch

MODEL_PATH = "/root/models/esm2_t33_650M_UR50D.pt"

# 直接读取权重
model_data = torch.load(MODEL_PATH, map_location='cuda')

# model_data 是一个字典或对象，取决于你的 esm 版本
print(type(model_data))
print(model_data.keys())  # 查看里面有哪些字段
