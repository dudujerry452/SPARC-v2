from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__, template_folder='templates')
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')


@app.route('/')
def index():
    return render_template('index.html')


def run_server(host='0.0.0.0', port=5000):
    socketio.run(app, host=host, port=port)
