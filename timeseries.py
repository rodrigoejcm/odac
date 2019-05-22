from skmultiflow.data.regression_generator import RegressionGenerator

###################################
## Timeseries class
###################################

class Timeseries:

    def __init__(self, name, n_samples, random_state):
        self.name = name
        self.current_value = None
        self.strem_data_generator = RegressionGenerator(n_samples=n_samples+1, n_features=1, n_informative=2, n_targets=1, random_state=random_state)
        self.strem_data_generator.prepare_for_use()
        self.next_val()
        #print("TS ", self.name , " inicializado com valor: ", self.current_value )

    def next_val(self):
        self.current_value = float('%.2f'%(self.strem_data_generator.next_sample()[0][0][0]))
        return self.current_value

####################################

def generate_multi_timeseries(n_ts, n_samples, random_state):
    dic_of_timeseries = {}
    for i in range(0,n_ts):
        dic_of_timeseries["S"+str(i)] = Timeseries(name="S"+str(i), n_samples=n_samples, random_state=random_state+i)
    return dic_of_timeseries

def get_timeseries_next_value(dic_of_timeseries):
    next_values = []
    for key, value in dic_of_timeseries.items():
        next_values.append((key,value.next_val()))
    return next_values
