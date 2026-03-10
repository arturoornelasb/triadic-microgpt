import sys, torch
sys.path.insert(0, '.')
from src.evaluate import load_model
from src.chat import generate_response

model, tok, cfg = load_model(
    'checkpoints/torch_run15_strongalign/model_L12_D512_B64_best.pt',
    'checkpoints/torch_run15_strongalign/tokenizer.json',
    'cpu'
)
resp, h_p, h_r = generate_response(model, tok, "Once upon a time", torch.device('cpu'))
print("Response:", resp[:150])
print("h_prompt shape:", h_p.shape)
print("h_resp shape:", h_r.shape)

# Verify distinct Phi values for different words
import torch
from src.triadic import PrimeMapper
pm = PrimeMapper(n_bits=64)
words = ["king", "queen", "dog"]
for w in words:
    ids = torch.tensor(tok.encode(w), dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        _, h, _ = model(ids)
    bits = (torch.tanh(model.triadic_head(model.ln_f(h[:, -1, :]))) > 0).squeeze().tolist()
    phi = pm.bits_to_phi(bits)
    print(f"  {w}: Phi={phi}")
