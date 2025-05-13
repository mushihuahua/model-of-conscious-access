import os
import json
import gzip

import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class Cache:

    def __init__(self, path='./.cache/'):
        self.path = path

        if not os.path.exists(path):
            os.makedirs(path)

    def cache_figure_data(self, data: dict, figure: str):

        path = f"{self.path}/{figure}.json"

        with gzip.open(path, "w") as f:
            f.write(json.dumps(data, cls=NumpyEncoder).encode('utf-8'))
            

    def get_figure_data(self, figure: str):

        path = f"{self.path}/{figure}.json"
        data = {}

        if(os.path.exists(path)):
            with gzip.open(path, 'r') as f:
                _data = f.read().decode('utf-8')
                data = json.loads(_data)
        
        else:
            print(f"{path} figure data does not exist, be sure to run the simulation first.")

        return data