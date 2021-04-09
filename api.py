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

    return 'Score for face: {:.4f}'.format(faces_pred[0])


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
        'uploads/videoplayback_1.mp4')

    im_real_face = vid_real_faces[0]['faces'][0]

    faces_real_t = torch.stack([transf(image=frame['faces'][0])[
                               'image'] for frame in vid_real_faces if len(frame['faces'])])

    with torch.no_grad():
        faces_real_pred = net(faces_real_t.to(device)).cpu().numpy().flatten()

    return 'Average score for REAL video: {:.4f}'.format(
        expit(faces_real_pred.mean()))


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/form')
def form():
    return render_template('form.html')


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


app.run(host='localhost', port=5000, debug=True)
