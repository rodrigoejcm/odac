from skmultiflow.data.regression_generator import RegressionGenerator

###################################
## Timeseries class
###################################

class Timeseries:

    current_value = None
    strem_data_generator = RegressionGenerator(n_samples=50, n_features=1, n_informative=2, n_targets=1, random_state=2)
    strem_data_generator.prepare_for_use()

    def __init__(self, name):
        self.name = name
        self.next_val()
        #print("TS ", self.name , " inicializado com valor: ", self.current_value )
    
    def next_val(self):
        self.current_value = float('%.4f'%(self.strem_data_generator.next_sample()[0][0][0]))
        return self.current_value

####################################

def generate_multi_timeseries(n):
    dic_of_timeseries = {}
    for i in range(0,n):
        dic_of_timeseries["S"+str(i)] = Timeseries(name="S"+str(i))
    return dic_of_timeseries

def get_timeseries_next_value(dic_of_timeseries):
    next_values = []
    for key, value in dic_of_timeseries.items():
        next_values.append((key,value.next_val()))
    return next_values




