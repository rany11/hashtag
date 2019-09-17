from flask import Flask, request

app = Flask(__name__, static_url_path='/static')


@app.route('/')
def index():
    return 'pasten'


@app.route('/image/<id>')
def image(id):
    return app.send_static_file('banana.jpg')


@app.route('/next')
def next():
    return '1'


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
    return 'got it'


if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
