import os

from flask import Flask, request, render_template

from inference import predict_flower

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods = ['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        if 'flower_img' not in request.files :
            print("Please upload the Image")
            return
        imageSave = request.files['flower_img']
        imageSave.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload.jpg'))
        flower_name = predict_flower()
        imagename = os.path.join(app.config['UPLOAD_FOLDER'], 'upload.jpg')
        return render_template('result.html', flower= flower_name, im_name = imagename)

if __name__ == '__main__':
    app.run(debug='False', port=os.getenv('PORT', 5000))