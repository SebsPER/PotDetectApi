import os 
from flask import Flask, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
import sys
import cv2
import importlib.util
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter
import numpy as np
import json
import time
import base64

UPLOAD_FOLDER = './img'
ALLOWED_EXTENSIONS = set(['jpg', 'JPG', 'JPEG', 'jpeg', 'png'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def tflite_detect_images(modelpath, image_path, lblpath, min_conf=0.5, savepath='/content/results', txt_only=False):
    start_time = time.time()
    # Load the label map into memory
    with open(lblpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the Tensorflow Lite model into memory
    #interpreter = tf.lite.Interpreter(model_path=modelpath)
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    float_input = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Load image and resize to expected shape [1xHxWx3]
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)
    
    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std
        
    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    end_time = time.time()
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(start_time)
    print(end_time)
    print("Elapsed time: ", elapsed_time) 
    
    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects

    # return json_box, json_cla, json_sco
    det_labels = []
        # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))

            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            det_labels.append(object_name)
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

    #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #edoc = str(elapsed_time)
    _, im_arr = cv2.imencode('.jpg', image)  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    #cv2.imwrite('./new_img/proc_im'+edoc.replace(".", "_")+'.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])

    lbl_count = {}
    for i in det_labels:
        if i not in lbl_count.keys():
            lbl_count[i] = 1
        else:
            lbl_count[i] += 1
    
    for i in labels:
        if i not in lbl_count.keys():
            lbl_count[i] = 0

    return lbl_count, elapsed_time, im_b64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/media/upload', methods=['POST'])
def upload_media():
    print(request.files)
    if 'file' not in request.files:
        return jsonify({'error': 'media not provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'no file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print(filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        lbl, elapsed, baseImg = tflite_detect_images('./models/detect.tflite', './img/'+filename, './models/labelmap.txt', 0.2)
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        proc_img = str(baseImg)
    return jsonify({'Hueco':lbl['Hueco'],'HuecoGrave': lbl['HuecoGrave'], 'Grieta': lbl['Grieta'], 'elapsed': elapsed, 'base_64':proc_img[2:len(proc_img)-1]})

@app.route('/get_img/<path:filename>', methods=['GET'])
def get_img(filename):
    #file_id = request.args['id']
    print(filename)
    proc_id = filename.replace('.', '_')
    print(proc_id)
    print(proc_id[0:len(proc_id)-4])
    #print('./new_img/proc_im'+proc_id+'.jpg')
    return send_from_directory(directory='./new_img/' ,path='proc_im'+proc_id[0:len(proc_id)-4]+'.jpg', as_attachment=False)

@app.route('/del_img', methods=['DELETE'])
def del_img():
    file_id = request.args['id']
    proc_id = file_id.replace('.', '_')
    os.remove('./new_img/proc_im'+proc_id+'.jpg')
    return jsonify({'msg':'file removed'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)