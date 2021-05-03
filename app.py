import sys
from scipy.special import expit
from architectures import fornet, weights
from isplutils import utils
from blazeface import FaceExtractor, BlazeFace, VideoReader
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import torch
from torch.utils.model_zoo import load_url
from PIL import Image
import matplotlib.pyplot as plt
from flask_cors import CORS, cross_origin
from flask import jsonify
from twitterdl import TwitterDownloader

# """
# Choose an architecture between
# - EfficientNetB4
# - EfficientNetB4ST
# - EfficientNetAutoAttB4
# - EfficientNetAutoAttB4ST
# - Xception
# """
# net_model = 'EfficientNetAutoAttB4'

# """
# Choose a training dataset between
# - DFDC
# - FFPP
# """
# train_db = 'DFDC'

# device = torch.device(
#     'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# face_policy = 'scale'
# face_size = 224
# frames_per_video = 32
# model_url = weights.weight_url['{:s}_{:s}'.format(net_model, train_db)]
# net = getattr(fornet, net_model)().eval().to(device)
# net.load_state_dict(load_url(model_url, map_location=device, check_hash=True))
# facedet = BlazeFace().to(device)
# facedet.load_weights("./blazeface/blazeface.pth")
# facedet.load_anchors("./blazeface/anchors.npy")
# face_extractor = FaceExtractor(facedet=facedet)
# videoreader = VideoReader(verbose=False)


# def video_read_fn(x): return videoreader.read_frames(
#     x, num_frames=frames_per_video)


# transf = utils.get_transformer(
#     face_policy, face_size, net.get_normalizer(), train=False)


def predictImage(image):
    image = Image.open(image)
    image = face_extractor.process_image(img=image)
    # take the face with the highest confidence score found by BlazeFace
    image = image['faces'][0]

    faces_t = torch.stack([transf(image=im)['image']
                           for im in [image]])

    with torch.no_grad():
        faces_pred = torch.sigmoid(
            net(faces_t.to(device))).cpu().numpy().flatten()

    d = {'result': str(faces_pred[0])}

    return jsonify(d)


"""
Choose an architecture between
- EfficientNetB4
- EfficientNetB4ST
- EfficientNetAutoAttB4
- EfficientNetAutoAttB4ST
- Xception
"""
net_model = 'EfficientNetAutoAttB4'

"""
Choose a training dataset between
- DFDC
- FFPP
"""
train_db = 'DFDC'

device = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
face_policy = 'scale'
face_size = 224
frames_per_video = 32

model_url = weights.weight_url['{:s}_{:s}'.format(net_model, train_db)]
net = getattr(fornet, net_model)().eval().to(device)
net.load_state_dict(load_url(model_url, map_location=device, check_hash=True))

facedet = BlazeFace().to(device)
facedet.load_weights("./blazeface/blazeface.pth")
facedet.load_anchors("./blazeface/anchors.npy")
videoreader = VideoReader(verbose=False)
transf = utils.get_transformer(
    face_policy, face_size, net.get_normalizer(), train=False)


def video_read_fn(x): return videoreader.read_frames(
    x, num_frames=frames_per_video)


face_extractor = FaceExtractor(video_read_fn=video_read_fn, facedet=facedet)


def predictVideo(video):
    vid_real_faces = face_extractor.process_video(
        video)

    im_real_face = vid_real_faces[0]['faces'][0]

    faces_real_t = torch.stack([transf(image=frame['faces'][0])[
                               'image'] for frame in vid_real_faces if len(frame['faces'])])

    with torch.no_grad():
        faces_real_pred = net(faces_real_t.to(device)).cpu().numpy().flatten()

    d = {'result': str(faces_real_pred.mean())}
    print(d)
    return jsonify(d)


app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/link', methods=['POST', 'GET'])
def link():
    if request.method == 'POST':
        link = request.get_json()['link']
        tw = TwitterDownloader(link)
        fname = tw.download()
        # f = request.files['File']
        # # f.save(secure_filename(f.filename))
        # fname = os.path.join(
        #     app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        # f.save(fname)
        return predictVideo(fname)
        # return jsonify({"status": fname})


@app.route('/uploadI', methods=['POST', 'GET'])
def uploadI():
    if request.method == 'POST':
        print(request.files)
        f = request.files['File']
        # f.save(secure_filename(f.filename))
        fname = os.path.join(
            app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        f.save(fname)
        return predictImage(fname)


@app.route('/uploadV', methods=['POST', 'GET'])
def uploadV():
    if request.method == 'POST':
        print(request.files)
        f = request.files['File']
        # f.save(secure_filename(f.filename))
        fname = os.path.join(
            app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        f.save(fname)
        return predictVideo(fname)


@app.route('/upload', methods=['POST', 'GET'])
@cross_origin()
def upload():
    if request.method == 'POST':
        print(request.files)
        f = request.files['File']
        # f.save(secure_filename(f.filename))
        fname = os.path.join(
            app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        # f.save(fname)
        return 'ok'

    if request.method == "GET":
        d = {'name': 'ramu'}
        return jsonify(d)


@app.route("/", methods=['GET'])
def hello():
    return 'The API is UP'


# if __name__ == "__main__":
#     app.run(host='0.0.0.0')

# Tw = TwitterDownloader("https://twitter.com/i/status/1388581577714720772")
# print(Tw.download())


app.run(host='localhost', port=3000, debug=True)
