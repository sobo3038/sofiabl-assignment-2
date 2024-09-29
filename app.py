from flask import Flask, jsonify, request, send_file
from kmeans import KMeans
import numpy as np

app = Flask(__name__)
kmeans = None
dataset = None

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/generate', methods=['GET'])
def generate_dataset():
    global dataset
    dataset = np.random.randn(300, 2) * 10  # Generating a larger spread
    return jsonify({'dataset': dataset.tolist()})


@app.route('/step', methods=['POST'])
def step():
    global kmeans, dataset
    data = request.json
    k = int(data['k'])
    init_method = data['init_method']
    
    if kmeans is None:
        kmeans = KMeans(dataset, k, init_method)
    
    kmeans.step()
    return jsonify(kmeans.get_plot_data())

@app.route('/run', methods=['POST'])
def run_to_convergence():
    global kmeans, dataset
    data = request.json
    k = int(data['k'])
    init_method = data['init_method']
    
    kmeans = KMeans(dataset, k, init_method)
    kmeans.run_to_convergence()
    
    return jsonify(kmeans.get_plot_data())

@app.route('/reset', methods=['POST'])
def reset():
    global kmeans
    kmeans = None
    return jsonify({})

@app.route('/set_manual_centroids', methods=['POST'])
def set_manual_centroids():
    global kmeans, dataset
    data = request.json
    manual_centroids = np.array(data['centroids'])
    
    k = len(manual_centroids)
    kmeans = KMeans(dataset, k, init_method='manual', manual_centroids=manual_centroids)
    
    return jsonify(kmeans.get_plot_data())

@app.route('/set_manual_centroids_and_run', methods=['POST'])
def set_manual_centroids_and_run():
    global kmeans, dataset
    data = request.json
    manual_centroids = np.array(data['centroids'])
    
    k = len(manual_centroids)
    kmeans = KMeans(dataset, k, init_method='manual', manual_centroids=manual_centroids)
    kmeans.run_to_convergence()
    
    return jsonify(kmeans.get_plot_data())

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)

