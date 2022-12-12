import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from src import model
def get_harmonized(composite_image,fg_mask,save_dir='harmonized'):
    cuda = torch.cuda.is_available()
    harmonizer = model.Harmonizer()
    if cuda:
        harmonizer.load_state_dict(torch.load('pretrained/harmonizer.pth', map_location=torch.device('cuda')), strict=True)
        harmonizer = harmonizer.cuda()
    else:
        harmonizer.load_state_dict(torch.load('pretrained/harmonizer.pth', map_location=torch.device('cpu')), strict=True)
    harmonizer.eval()
    _comp = tf.to_tensor(composite_image)[None, ...]
    _mask = tf.to_tensor(fg_mask)[None, ...]
    _comp = _comp.cuda()
    _mask = _mask.cuda()
    with torch.no_grad():
        arguments = harmonizer.predict_arguments(_comp, _mask)
        _harmonized = harmonizer.restore_image(_comp, _mask, arguments)

    output_img = tf.to_pil_image(_harmonized.squeeze())
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    output_img.save(os.path.join(save_dir, 'harmonized.jpg'))

if __name__ == '__main__':
    composite_image = Image.open('data/composite.jpg')
    fg_mask = Image.open('data/fg_mask.jpg')
    get_harmonized(composite_image,fg_mask)
