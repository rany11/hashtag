from flask import (
    Flask, request, redirect, url_for,
    render_template
)
import requests

# Create the application instance
app = Flask(__name__, template_folder='templates', static_url_path='/static')

BACKEND_URL = 'http://127.0.0.1:8080'

# Create a URL route in our application for "/"
@app.route('/')
def home():
    """
    This function just responds to the browser ULR
    localhost:5000/

    :return:        the rendered template 'home.html'
    """
    fid = requests.get(BACKEND_URL + '/next').content.decode()
    hint = requests.get(BACKEND_URL + '/hint/' + fid).content.decode()
    accuracy = requests.get(BACKEND_URL + '/accuracy').content.decode()
    return render_template('home.html', fid=fid, hint=hint, accuracy=accuracy)


@app.route('/hashtag')
def hashtag():
    """
    This function just responds to the browser ULR
    localhost:5000/

    :return:        the rendered template 'home.html'
    """
    fid = requests.get(BACKEND_URL + '/next').content.decode()
    hint = requests.get(BACKEND_URL + '/hint/' + fid).content.decode()
    accuracy = requests.get(BACKEND_URL + '/accuracy').content.decode()
    return render_template('hashtag/index.html', fid=fid, hint=hint, accuracy=accuracy)


@app.route('/image/<fid>')
def image(fid):
    return requests.get(BACKEND_URL + '/image/' + fid).content


@app.route('/image')
def get_image():
    fid = requests.get(BACKEND_URL + '/next').content.decode()
    return image(fid)


NO_LABEL = 'no_label'
NO_ID = -1


@app.route('/oracle')
def oracle():
    sample_id = request.args.get('id', default=NO_ID, type=int)
    label = request.args.get('tag', default=NO_LABEL, type=str)

    print(requests.get(BACKEND_URL + '/oracle', params={'id': sample_id, 'tag': label}).status_code)

    return redirect(url_for('home'))


# If we're running in stand alone mode, run the application


if __name__ == '__main__':
    app.run(debug=True, port=8000, host='0.0.0.0')
