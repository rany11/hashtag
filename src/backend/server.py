from flask import Flask, request

from src.backend.active_learning.experimentation.dataset.loader import Loader
from PIL import Image
import os
import random
import string

from src.backend.session import Session

app = Flask(__name__, static_url_path='/static')
session = Session('mnist')


@app.route('/')
def index():
    return 'pasten'


@app.route('/image/<fid>')
def image(fid):
    fid = int(fid)
    data, _ = Loader.load_dataset('mnist')
    sample = data[fid % len(data)].reshape(8, 8) * 16
    im = Image.fromarray(sample).resize((256, 256)).convert("L")
    name = 'tmp' + ''.join((random.choice(string.ascii_lowercase) for _ in range(10))) + '.jpg'
    im.save(os.path.join(app.static_folder, name))
    return app.send_static_file(name)


@app.route('/next')
def next_image():
    return str(session.next())


@app.route('/sanity')
def sanity():
    print('becoming insane')
    return '2'


NO_LABEL = 'no_label'
NO_ID = -1


@app.route('/oracle')
def oracle():
    sample_id = request.args.get('id', default=NO_ID, type=int)
    label = request.args.get('tag', default=NO_LABEL, type=str)
    print(f'{sample_id} -> {label}')
    session.take_label(sample_id, label)
    return 'got it'


@app.route('/hint/<fid>')
def hint(fid):
    return str(session.predict(sample_id=fid))


if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
