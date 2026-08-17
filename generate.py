import torch
from gpt import GPT, encode, decode, device

model = GPT()
model.load_state_dict(torch.load('gpt-shakespeare.pt', map_location=device))
model = model.to(device)
model.eval()

with torch.no_grad():
    context = torch.zeros((1, 6), dtype=torch.long, device=device)
    context = context + torch.tensor(encode('JULIET'), dtype=torch.long, device=device)
    for top_k in [None, 10, 25]:
        print(f"\n{'='*20} T = 1.3, top_k = {top_k} {'='*20}")
        out = model.generate(context, max_new_tokens=300, temperature=1.3, top_k=top_k)
        print(decode(out[0].tolist()))