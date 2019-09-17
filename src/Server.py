from flask import Flask

app = Flask(__name__)


@app.route('/')
def index():
    return 'pasten'


@app.route('/hello')
def hello():
    return 'hello world'
