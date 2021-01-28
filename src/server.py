from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from io import BytesIO
import fire
import os
import sys
import numpy as np
import tensorflow as tf

import model, sample, encoder

conversation="""
you: hi
her: hey
you: i'm a human
her: i'm a robot
you: you ready?
her: yes :)
you: ok let's start chatting
her: sure, what do you want to talk about?"""

def interact_model(
    message="",
    model_name='1558M',
    models_dir='models',
    seed=None,
    length=20,
    temperature=1,
    top_k=0
):
    #models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    global conversation
    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
    with tf.Session(graph=tf.Graph()) as sess:
        np.random.seed(seed)
        tf.set_random_seed(seed)
        context = tf.placeholder(tf.int32, [1, None]) #isn't exists in other model
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=1,
            temperature=temperature, top_k=top_k
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        conversation = conversation + "\nyou: " + message
        conversation = conversation + "\nher: "
        sys.stdout.write("her: ")
        sys.stdout.flush()
        
        encoded_conversation = enc.encode(conversation)
        result = sess.run(output, feed_dict={
            context: [encoded_conversation]
        })[:, len(encoded_conversation):]
        text = enc.decode(result[0])
        
        #sys.stderr.write("=============="+text+"=================")
        #sys.stderr.flush()

        splits = text.split('\n')
        #line = splits[1] if len(splits)>1 else splits[0]
        #parts = line.split(': ')
        #reply = parts[1] if len(parts)>1 else parts[0]
        reply = splits[0]
        sys.stdout.write(reply+'\n')
        sys.stdout.flush()
        conversation = conversation + reply
        print(conversation)
        return reply

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        self.send_response(200)
        self.end_headers()
        response = BytesIO()
        data = json.loads(body.decode("utf-8"))
        result = ""
        if 'text' in data:
            result = interact_model(data['text'])
            exit_if=True
        response.write(result.encode())
        self.wfile.write(response.getvalue())


httpd = HTTPServer(('0.0.0.0', 8000), SimpleHTTPRequestHandler)
httpd.serve_forever()