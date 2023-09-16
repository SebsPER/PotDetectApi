import os 
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import sys
import cv2
import importlib.util
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import json
import time
import base64

UPLOAD_FOLDER = './img'
ALLOWED_EXTENSIONS = set(['jpg', 'JPG', 'JPEG', 'jpeg', 'png'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()

def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def display(imgs, pred, names, files, pprint=False, show=False, save=False, crop=False, render=False):
    
    lbl_count = {}

    for i, (im, pred) in enumerate(zip(imgs, pred)):
        str = f'image {i + 1}/{len(pred)}: {im.shape[0]}x{im.shape[1]} '
        if pred is not None:
            for c in pred[:, -1].unique():
                n = (pred[:, -1] == c).sum()  # detections per class
                str += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                lbl_count[names[int(c)]] = n.item()
            if show or save or render or crop:
                for *box, conf, cls in pred:  # xyxy, confidence, class
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(box, im, label=label, color=colors(cls))

        im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
        print(str.rstrip(', '))
        f = files[i]
        #im.save('./img') 
        opencvImage = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        _, im_arr = cv2.imencode('.jpg', opencvImage)  # im_arr: image in Numpy one-dim array format.
        im_bytes = im_arr.tobytes()
        im_b64 = base64.b64encode(im_bytes)

    for i in names.values():
        if i not in lbl_count.keys():
            lbl_count[i] = 0
    
    return lbl_count, im_b64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/media/upload', methods=['POST'])
def upload_media():
    #print(request.files)
    if 'file' not in request.files:
        return jsonify({'error': 'media not provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'no file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        #print(filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/yolov5_grieta.pt')  # local model
        im1 = cv2.imread('./img/'+filename)[..., ::-1]
        results = model(im1) # batch of images
        lbl, baseImg = display(results.ims, results.pred, results.names, results.files, save=True)
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        proc_img = str(baseImg)
    return jsonify({'Hueco':lbl['Hueco'],'HuecoGrave': lbl['HuecoGrave'], 'Grieta': lbl['Grieta'], 'elapsed': results.t[1], 'base_64':proc_img[2:len(proc_img)-1]})

if __name__ == '__main__':
    app.run(debug=True, port=5000)