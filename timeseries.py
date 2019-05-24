from skmultiflow.data.regression_generator import RegressionGenerator
import numpy as np

class SinGenerator:
    """Generator for a sine wave, with 10% random noise"""

    def __init__(self, start = 0, inc = 0.1, noise = 0.02):
        self.state = start
        self.inc = inc
        self.noise = noise

    def next_val(self):
        self.state = self.state + self.inc
        return np.sin(self.state) + np.random.randn()*self.noise

class CosGenerator:
    """Generator for a cosine wave, with 10% random noise"""

    def __init__(self, start = 0, inc = 0.1, noise = 0.02):
        self.state = start
        self.inc = inc
        self.noise = noise

    def next_val(self):
        self.state = self.state + self.inc
        return np.cos(self.state) + np.random.randn()*self.noise

class Timeseries:
    """A single timeseries, with an associated id, name and generator"""

    def __init__(self, name, generator):
        self.id = id
        self.name = name
        self.current_value = None
        self.generator = generator
        #self.file = open(self.name + ".csv","w")
        self.next_val()

    def next_val(self):
        self.current_value = self.generator.next_val()
        #self.file.write("{}\r\n".format(self.current_value))
        return self.current_value
