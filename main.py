from flask import json
from flask import Response
from flask import Flask
from flask import request
import joblib
import numpy as np

APP = Flask(__name__)
STATUS = "ok"
#CORS(APP)
@APP.route("/uploadContext", methods=['POST'])
def post_json_context():
    #labels_dict = {}
    response_dict = {}
    filename = "rf_mode.sav"
    try:
        context = request.get_json()
        print(context)
        loaded_model = joblib.load(filename)
        type_of_services = context["typeOfServices"] 
        row = np.array([])
        input = []
        for id, n in context["services"].items():
            input = np.append(np.array([input]),n)
        print(input)
        row = np.append(row, input)
        row = row.reshape(1,-1)
        label = loaded_model.predict(row)
        print(label)
        if context["typeOfServices"] == 1:
            response_dict['numberOfClasses'] = 2
            response_dict['classes'] = [0,20]
        else:
            response_dict['numberOfClasses'] = 4
            response_dict['classes'] = [0,20,200,1000]
        js_dump = json.dumps(response_dict)
        resp = Response(js_dump,status=200,
                        mimetype='application/json')
    except RuntimeError as err:
        response_dict = {'error': 'error occured on server side. Please try again'}
        js_dump = json.dumps(response_dict)
        resp = Response(js_dump, status=500,
                        mimetype='application/json')
    return resp
if __name__ == '__main__':
    APP.run(host='0.0.0.0', port=5000)