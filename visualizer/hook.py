import threading

from .server import run_server, socketio
from .plotter import tensor_to_image

_server_started = False
_server_lock = threading.Lock()


def start_visualizer(host='0.0.0.0', port=5000):
    """Start the Flask-SocketIO visualizer server in a background thread."""
    global _server_started
    with _server_lock:
        if _server_started:
            return
        _server_started = True

    thread = threading.Thread(
        target=run_server,
        kwargs={'host': host, 'port': port},
        daemon=True
    )
    thread.start()


def send_tensor(name, tensor, title=None):
    """Convert a tensor to an image and broadcast it to all connected clients."""
    if title is None:
        title = name
    image = tensor_to_image(tensor, title=title)
    socketio.emit('tensor', {'name': name, 'image': image})
