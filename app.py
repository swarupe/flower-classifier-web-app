import os

from flask import Flask, request, render_template

from inference import predict_flower

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        if 'flower_img' not in request.files :
            print("Please upload the Image")
            return
        image = request.files['flower_img']
        # image.save('uploaded_image.jpeg') # Saving image to disk
        image_bytes = image.read(image_bytes)
        flowe_name = predict_flower(image_bytes)
        return render_template('result.html', flower= flowe_name)

if __name__ == '__main__':
    app.run(debug='False', port=os.getenv('PORT', 5000))