from skmultiflow.data.regression_generator import RegressionGenerator
import numpy as np

class SinGenerator:
    """Generator for a sine wave, with 10% random noise"""

    def __init__(self, start = 0, inc = 0.1):
        self.state = start
        self.inc = inc

    def next_val(self):
        self.state = self.state + self.inc
        return np.sin(self.state) + np.random.randn()/10

class CosGenerator:
    """Generator for a cosine wave, with 10% random noise"""

    def __init__(self, start = 0, inc = 0.1):
        self.state = start
        self.inc = inc

    def next_val(self):
        self.state = self.state + self.inc
        return np.cos(self.state) + np.random.randn()/10

class Timeseries:
    """A single timeseries, with an associated id, name and generator"""

    def __init__(self, id, name, generator):
        self.id = id
        self.name = name
        self.current_value = None
        self.generator = generator
        self.next_val()

    def next_val(self):
        self.current_value = self.generator.next_val()
        return self.current_value

####################################

def generate_multi_timeseries(n_ts):
    """Generates the specified number of timeseries. All timeseries with an odd id
       are based on a sine wave and all timeseries with an even id on a cosine wave"""
    dic_of_timeseries = {}
    for i in range(0,n_ts):
        name = "S"+str(i)
        print("{}: {} generator".format(name, "sine" if i%2==0 else "cosine"))
        dic_of_timeseries[name] = Timeseries( \
            id=i, \
            name=name, \
            generator = SinGenerator() if i%2==0 else CosGenerator())
    return dic_of_timeseries

def get_timeseries_next_value(dic_of_timeseries):
    next_values = []
    for key, value in dic_of_timeseries.items():
        next_values.append((key,value.next_val()))
    return next_values
