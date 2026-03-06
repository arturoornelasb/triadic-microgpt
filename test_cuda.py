import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
    x = torch.randn(1000, 1000, device='cuda')
    y = x @ x.T
    print(f"GPU compute test: {y.shape} tensor OK")
