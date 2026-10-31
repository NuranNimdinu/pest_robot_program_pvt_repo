from flask import Flask, render_template
import time
import requests
import threading

flask_server_ready = False

# Initialize Flask
app = Flask(__name__)

@app.route('/health')
def health():
    return {"status": "ready"}, 200

@app.route('/')
def serve_app():
    # render_template automatically looks inside the "templates" folder
    return render_template('index.html')

def get_server_status():
    return flask_server_ready

def wait_for_server_ready(url="http://127.0.0.1:5000/health", timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                print("Flask server is ready to receive requests!")
                return True
        except requests.exceptions.RequestException:
            # Server is still starting up or port is not yet open
            time.sleep(0.2)
    
    print("Timed out waiting for Flask server to start.")
    return False

def check_server_status():
    global flask_server_ready
    if wait_for_server_ready():
        flask_server_ready = True
    else:
        flask_server_ready = False

def run_web_app():
    server_ready_check_th = threading.Thread(
        target=check_server_status, 
        daemon=True
    )
    server_ready_check_th.start()

    # host='0.0.0.0' exposes it to your local network (e.g., your phone can connect)
    # app.run(host='0.0.0.0', port=5000, debug=True)
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)