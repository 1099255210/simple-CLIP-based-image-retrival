import gradio as gr

import torch
import clip
import os
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load('ViT-B/32', device=device)

data_location =  './imgs'
img_dict = {}
for inx, f in enumerate(os.listdir(data_location)):
  img_dict[inx] = f
img_nums = len(img_dict)

def fn_text(instr):
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

def fn_image(user_img):
  user_img_input = preprocess(user_img).unsqueeze(0).to(device)

  # user image encode
  with torch.no_grad():
    user_img_f = model.encode_image(user_img_input)
  user_img_f /= user_img_f.norm(dim=-1, keepdim=True)

  sim = {}

  for i in range(img_nums):
    
    image_path = f'{data_location}/{img_dict[i]}'
    img = Image.open(image_path)
    img_input = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
      img_f = model.encode_image(img_input)

    img_f /= img_f.norm(dim=-1, keepdim=True)
    similarity = 100 * img_f @ user_img_f.T
    sim[i] = similarity

  res = sorted(sim.items(), key=lambda s:s[1], reverse=True)
  retval = [ f'{data_location}/{img_dict[res[i][0]]}' for i in range(3) ]
  return retval

# css_output = '.object-contain {height: 100px !important}'
# demo = gr.Interface(
#   fn = fn_text,
#   inputs = 'text',
#   outputs = [gr.Image(type='file', label=None) for _ in range(3)],
#   css = css_output,
# )

with gr.Blocks() as demo:
  gr.Markdown('Search for images based on text or similar image as clue.')
  with gr.Tab('Text'):
    with gr.Row():
      text_input = gr.Textbox()
      text_output = [gr.Image(type='file', label=None) for _ in range(3)]
      text_button = gr.Button('Search')
  with gr.Tab('Image'):
    with gr.Row():
      image_input = gr.Image(type='pil')
      image_output = [gr.Image(type='file', label=None) for _ in range(3)]
    image_button = gr.Button('Search')

  text_button.click(fn_text, inputs=text_input, outputs=text_output)
  image_button.click(fn_image, inputs=image_input, outputs=image_output)

demo.launch()