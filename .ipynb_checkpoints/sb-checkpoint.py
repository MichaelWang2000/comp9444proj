import torch
import torch.nn as nn

print(torch.cuda.device_count())

if torch.cuda.is_available():
    # 获取可用GPU的数量
    num_devices = torch.cuda.device_count()
    device_ids = list(range(num_devices))
    print(f"Using {num_devices} GPU(s): {device_ids}")
    
    # 创建模型并使用DataParallel
    #network = nn.DataParallel(network, device_ids=device_ids)
else:
    print("CUDA is not available. Training on CPU.")