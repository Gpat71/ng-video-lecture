import torch

from gpt import GPTLanguageModel
from gpt import device
from gpt import decode

# Load the model (instead of training)

model = GPTLanguageModel()
m = model.to(device)
m.load_state_dict(torch.load("gpt_model.pth", map_location=device))
m.eval()  # set to evaluation mode

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
# open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
