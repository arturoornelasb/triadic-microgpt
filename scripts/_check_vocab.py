import torch
ckpt = torch.load('checkpoints/torch_run15_strongalign/model_L12_D512_B64_best.pt', map_location='cpu', weights_only=False)
cfg = ckpt.get('config', {})
print('Config:', cfg)
state = ckpt.get('model_state_dict', ckpt)
for k, v in state.items():
    if 'wte' in k or 'embed' in k.lower():
        print(f'{k}: {v.shape}')
