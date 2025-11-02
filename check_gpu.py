"""
检查 GPU 可用性和 PyTorch 配置
"""

import torch

print("="*60)
print("PyTorch GPU 检查")
print("="*60)

# 检查 PyTorch 版本
print(f"\nPyTorch 版本: {torch.__version__}")

# 检查 CUDA 可用性
cuda_available = torch.cuda.is_available()
print(f"\nCUDA 可用: {cuda_available}")

if cuda_available:
    # CUDA 版本
    print(f"CUDA 版本: {torch.version.cuda}")
    
    # GPU 数量
    gpu_count = torch.cuda.device_count()
    print(f"GPU 数量: {gpu_count}")
    
    # 当前 GPU
    current_device = torch.cuda.current_device()
    print(f"当前 GPU ID: {current_device}")
    
    # GPU 名称
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # GPU 内存
    print(f"\nGPU 内存信息:")
    for i in range(gpu_count):
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i} 总内存: {total_memory:.2f} GB")
    
    # 测试 GPU 运算
    print(f"\n测试 GPU 运算...")
    device = torch.device('cuda')
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    z = torch.mm(x, y)
    print(f"✓ GPU 运算测试成功！")
    
else:
    print("\n⚠️  GPU 不可用，将使用 CPU 训练")
    print("\n如果您有 NVIDIA GPU，请安装 CUDA 版本的 PyTorch:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu118")

# 推荐系统使用的设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n推荐系统将使用设备: {device}")

print("\n" + "="*60)
