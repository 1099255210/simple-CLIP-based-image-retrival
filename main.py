import gradio as gr

import torch
import clip
import os
from PIL import Image
from IPython.display import display
from tqdm.notebook import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

data_location =  "./text_imgs"
img_dict = {}
for inx, f in enumerate(os.listdir(data_location)):
  img_dict[inx] = f
img_nums = len(img_dict)


def fn(instr):
  text_input = clip.tokenize(instr).to(device)
  with torch.no_grad():
    text_f = model.encode_text(text_input)
  text_f /= text_f.norm(dim=-1, keepdim=True)

  sim = {}

  for i in range(img_nums):
    
    image_path = f'{data_location}/{img_dict[i]}'
    img = Image.open(image_path)
    img_input = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
      img_f = model.encode_image(img_input)

    img_f /= img_f.norm(dim=-1, keepdim=True)
    similarity = 100 * img_f @ text_f.T
    sim[i] = similarity

  res = sorted(sim.items(), key=lambda s:s[1], reverse=True)
  retval = [ f'{data_location}/{img_dict[res[i][0]]}' for i in range(3) ]
  return retval

# css_output = ".output-image, .input-image, .image-preview {height: 100px !important}"
css_output = ".object-contain {height: 100px !important}"

demo = gr.Interface(
  fn = fn,
  inputs = 'text', 
  outputs = [gr.Image(type='file', label=None) for _ in range(3)],
  css = css_output,
)
demo.launch(enable_queue = True,)