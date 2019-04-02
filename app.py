from flask import request
import os
from flask import Flask,jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import worker as wrk


app = Flask(__name__)
CORS(app)


@app.route('/')
def main():
	return "Server is up and running"

@app.route('/pos',method=['POST'])
def find_position():
	rssi = request.form['rssi']
	position = wrk.calculate(rssi)
	return position



if __name__=="__main__":
    # app.run(debug = True)
	app.debug = True
	port = int(os.environ.get("PORT",5000))
	app.run(host='0.0.0.0',port = port)