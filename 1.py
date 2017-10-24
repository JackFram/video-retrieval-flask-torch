from flask import Flask
from flask import render_template
from flask import request
from flask import make_response
from werkzeug.utils import secure_filename
from flask import url_for
from flask import json
from flask import session
from flask import send_from_directory
import pickle
import os
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torch.autograd import Variable
import pandas as pd
import numpy as np
from model import pre_res_net
from PIL import Image
from data_saver import Data_saver
import subprocess
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/Users/zhangzhihao/Documents/webbrain/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

app.secret_key = 'slsajdjl@@###434sakd@!#@##'

# html = '''
#     <!DOCTYPE html>
#     <title>Upload File</title>
#     <h1>图片上传</h1>
#     <form method=post enctype=multipart/form-data >
#          <input type=file name=file>
#          <input type=submit value=上传>
#     </form>
#     '''

# prepare the function to abtract feature from the picture
ffmpeg_1 = 'ffmpeg -i /Users/zhangzhihao/Documents/webbrain/data/clips/'
ffmpeg_2 = ' -vcodec copy -acodec copy -ss '
ffmpeg_3 = ' -t '
ffmpeg_4 = ' -y /Users/zhangzhihao/Documents/webbrain/uploads/upload2.mp4'
ffmpeg_5 = 'ffmpeg -i /Users/zhangzhihao/Documents/webbrain/uploads/upload2.mp4 -vcodec h264 /Users/zhangzhihao/Documents/webbrain/uploads/upload2.mp4 -y'
upload_url = '/Users/zhangzhihao/Documents/webbrain/uploads/upload2.mp4'
file_url = '/Users/zhangzhihao/Documents/webbrain/data/pkl_saver/feature.pkl'
data = pd.read_csv('/Users/zhangzhihao/Documents/webbrain/data/features/feature_matrix.csv')
saver = Data_saver(data, file_url)
fea_base = np.array(saver.load())
clips_time = {10: 641, 1: 670, 2: 678, 3: 727, 4: 960, 5: 873, 6: 992, 7: 871, 8: 969, 9: 881}


def time_tran(time):
    if time <= 10:
        return (0,10)
    else:
        return (int(time)-10,20)


def up(i):
    num = 0
    for m in range(i):
        num += clips_time[(m+9) % 10+1]
    return num


def video_time(index):
    if index+1 <= up(1):
        return (10, index*0.2)
    elif index+1 <= up(2) and index + 1 > up(1):
        return (1, (index-up(1))*0.2)
    elif index+1 <= up(3) and index + 1 > up(2):
        return (2, (index-up(2))*0.2)
    elif index+1 <= up(4) and index + 1 > up(3):
        return (3, (index-up(3))*0.2)
    elif index+1 <= up(5) and index + 1 > up(4):
        return (4, (index-up(4))*0.2)
    elif index+1 <= up(6) and index + 1 > up(5):
        return (5, (index-up(5))*0.2)
    elif index+1 <= up(7) and index + 1 > up(6):
        return (6, (index-up(6))*0.2)
    elif index+1 <= up(8) and index + 1 > up(7):
        return (7, (index-up(7))*0.2)
    elif index+1 <= up(9) and index + 1 > up(8):
        return (8, (index-up(8))*0.2)
    elif index+1 <= up(10) and index + 1 > up(9):
        return (9, (index-up(9))*0.2)
    return 0


def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    html = render_template('index.html')
    video_html = render_template('video.html')
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'pic.jpeg'))
            img = load_image(os.path.join(app.config['UPLOAD_FOLDER'], 'pic.jpeg'), transform)
            net = pre_res_net()
            fea = net(to_var(img)).squeeze().data.numpy()
            min = 1000
            min_index = 0
            for i in range(fea_base.shape[0]):
                dist = np.sqrt(np.sum(np.square(fea - fea_base[i, 1:])))
                if dist < min:
                    min = dist
                    min_index = i
            video = str(video_time(min_index)[0])+'.mp4'
            s, t = time_tran(video_time(min_index)[1])
            strcmd = ffmpeg_1 + video + ffmpeg_2 + str(s) + ffmpeg_3 + str(t) + ffmpeg_4
            if os.path.exists(upload_url):
                subprocess.call('rm '+upload_url, shell=True)
            subprocess.call(strcmd, shell=True)
            # subprocess.call(ffmpeg_5, shell=True)
            return video_html
    return html