from flask import Flask, render_template, request
import cv2
import numpy as np
import face_recognition
import os
# from flask_sqlalchemy  import SQLAlchemy

app = Flask(__name__)

# app.config["SQLALCHEMY_DATABASE_URI"]="sqlite:///data.db"
# db=SQLAlchemy(app)
# Load the face encodings
encodes = np.load('faces.dat')
classNames = []

# class database(db.Model):
#     sno=db.Column(db.Integer,primary_key=True)
#     namee=db.Cloumn(db.String(200))


for img in os.listdir('faces'):
    classNames.append(os.path.splitext(img)[0])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')

    name = request.form['name']
    print(f"Received data: Name={name}, File={file.filename}")

    # Save the uploaded image with the user's name
    filename = f'faces/{name}.jpg'
    file.save(filename)
    
    # data=database(namee=name)
    # db.session.add(data)
    # db.session.commit()

    return render_template('index.html', result=f"Image saved as {filename}")


if __name__ == '__main__':
    app.run(debug=True)
