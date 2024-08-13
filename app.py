import os
import numpy as np
import pandas as pd
from model import image_pre, predict 
from flask import Flask, render_template, request

app = Flask(__name__)

#UPLOAD_FOLDER = 'C:\Aaishni Study Courses\IIT Bombay courses\D-E placement courses\Data Science\Covid 19 detection using CT scans Project\Project\static'
#ALLOWED_EXTENSIONS = set(['png'])
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html',result='test result')

#@app.route('/',methods=['POST','GET'])
#def upload_file():
    result = None
    if 'file1' not in request.files:
        return 'there is no file1 in form!'
    file1 = request.files['file1']
    path = os.path.join(app.config['UPLOAD_FOLDER'],'input.png')
    file1.save(path)
    print("File saved at:", path)
    data = image_pre(path)
    s = predict(data)
    if s == 1:
        result = 'No Covid Detected'
    else:
        result = 'Covid Detected'
    return render_template('index.html',result=result or 'Processing failed.')

if __name__ == '__main__':
    app.run(debug=True,port=8000)