import json
from base64 import b64encode
import shutil

from urllib.request import urlopen
import requests
from flask import Flask

from scripts.print_predictions import parse_results

# "https://images.dog.ceo/breeds/beagle/n02088364_12154.jpg"

app = Flask(__name__)

@app.route('/test/{string}')
def test(string):
    return 'hello world'

@app.route('/predict/<breed>/<image_file>')
def route_request(breed, image_file):
    image_url = f"https://images.dog.ceo/breeds/{breed}/{image_file}"

    return get_results(image_url)


def get_results(image_url):
    service_uri = 'http://52.180.90.254/score'

    with urlopen(image_url) as response:
        with open('temp.jpg', 'bw+') as f:
            shutil.copyfileobj(response, f)

    def image_to_json(filename):
        with open(filename, 'rb') as f:
            content = f.read()
        base64_bytes = b64encode(content)
        base64_string = base64_bytes.decode('utf-8')
        raw_data = {'image': base64_string}
        return json.dumps(raw_data, indent=2)

    # Turn image into json and send an HTTP request to the prediction web service
    input_data = image_to_json('temp.jpg')
    headers = {'Content-Type':'application/json'}
    resp = requests.post(service_uri, input_data, headers=headers)

    # Extract predication results from the HTTP response
    result = resp.text.strip("}\"").split("[")
    return parse_results(result)
