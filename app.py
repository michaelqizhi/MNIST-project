
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import datetime
import os
from cassandra.cluster import Cluster
from PIL import Image
from tensorflow.python.saved_model import tag_constants


ALLOWED_EXTENSIONS = set(['png'])
KEYSPACE = "mnistkeyspace"

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def connect_cassandra():
    # cluster = Cluster([os.environ.get('CASSANDRA_PORT_9042_TCP_ADDR', 'localhost')], port=int(os.environ.get('CASSANDRA_PORT_9042_TCP_PORT', 9042)))
    cluster = Cluster(['zq-cassandra'])
    session = cluster.connect(keyspace=KEYSPACE)
    return session



@app.route('/', methods=['GET', 'POST'])
def upload_file():
    #user submit a form
    if request.method == 'POST':
        #In case no file was uploaded
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        #In case file has an empty name
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        #Everything is correct and we can run the prediction
        if file and allowed_file(file.filename):
            #save and read uploaded image
            filename = secure_filename(file.filename)
            file.save(secure_filename(file.filename))
            image = Image.open(file.filename)
            flatten_img = np.reshape(image, 784)

            #load model and run prediction

            scores = sess.run(output_tensor, {input_tensor: [flatten_img]})

            #store into cassandra
            current_time = str (datetime.datetime.now())
            session.execute("INSERT INTO mnistkeyspace.inputs (time,file_name,guess) VALUES (%s,%s,%s)",[current_time, filename, scores.argmax()])
            return 'My Guess is: '  + str (scores.argmax()) + '. Data has been added to database'
    return '''
    <!doctype html>
    <title>MNIST Project by Zhi Qi</title>
    <h1>Welcome to My Project</h1>
    <h4>*This Project is completed by Zhi Qi, with the assitance from Instructor Fan Zhang*<h4>
    <p>Please upload a image of a handwritten digit, and I will guess what that number is!<p>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Submit>
    </form>

    '''

if __name__ == "__main__":
    session = connect_cassandra()
    with tf.Session(graph=tf.Graph()) as sess:
        export_dir = ('savedModel')
        model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], export_dir)
        loaded_graph = tf.get_default_graph()

        # get necessary tensors by name
        input_tensor_name = model.signature_def['predict_images'].inputs['images'].name
        input_tensor = loaded_graph.get_tensor_by_name(input_tensor_name)
        output_tensor_name = model.signature_def['predict_images'].outputs['scores'].name
        output_tensor = loaded_graph.get_tensor_by_name(output_tensor_name)
        app.run(host='0.0.0.0', port=80)
