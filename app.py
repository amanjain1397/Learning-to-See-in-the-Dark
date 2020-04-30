import torch
import wget, os
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from utils.models import DarkDataset, UNet
from utils.models import pack_raw, resize_bayer

app = Flask(__name__)
network = UNet()
network.load_state_dict(torch.load('checkpoint_result/epoch_4000_512.pth.tar'))
device = torch.device("cuda:0" if torch.cuda.is_available()  else 'cpu')
network = network.to(device)
network.eval()

input_ = 'app_data\\input\\input.ARW'
output_ = 'app_data\\output\\output.jpg'

def produce(filename):
    
    input_ = pack_raw(filename)
    input_ = resize_bayer(input_, 30)
    input_ = torch.tensor(input_).to(device)
    input_ = input_.permute(2, 0, 1).unsqueeze(0)
    H, W = input_.shape[2:]

    output = network(input_)
    output = (np.clip(np.transpose(output.detach().squeeze(0).cpu().numpy(), (1,2,0)), 0., 1.) * 255).astype(np.uint8)
    img = Image.fromarray(output).resize((2 * W,2 * H), Image.BICUBIC)
    return img

@app.route('/')
def hello():
    return 'Hi, this is the front page of the this Project!'

@app.route('/evaluate', methods = ['POST'])
def evaluate():
    if request.method == 'POST':
        url = request.get_data().decode('utf_8')
        if os.path.exists(input_):
            os.remove(input_)
        wget.download(url, input_)

        img = produce(input_)
        img.save(output_)
        f = open(output_, 'r+b').read()
        return f

if __name__ == "__main__":
    app.run(debug=True)