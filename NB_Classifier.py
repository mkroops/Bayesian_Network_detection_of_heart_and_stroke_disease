#Author: Manigandan Sivalingam
#Description: Calculate probablistic query using Naive bayes algorithm

import sys
#tabulate table
from tabulate import tabulate

#Naive Bayes classifier
class NB_Classifier:
    random_variable = []
    no_of_random_variable = 0
    variable_keys = {}
    Training_values = []
    data_instances = 0
    predictor = ''
    estimated_parameters_prob = {}
    missing_value = 0.000001
    probablities = {}
    pred = []
    probablity_distribution = {}
    normalised_distribution = {}
    test_probablity_var_type = {}
    train_file = None
    Test_file = None

    def __init__(self):
        self.train_file = sys.argv[1]
        self.test_file = sys.argv[2]
    
    #read data from train file
    def read_data(self) :
        first_line_csv = True
        current_line_listed_format = []
        dict_key_initial_count = True
        with open(self.train_file) as csv_file:
            for line in csv_file:
                if first_line_csv == True:
                    #take random variable
                    self.random_variable = line.strip().split(',')
                    self.no_of_random_variable = len(self.random_variable)
                    self.predictor = self.random_variable[self.no_of_random_variable - 1]
                    first_line_csv = False
                else:
                    #update training values in dictionary
                    self.Training_values.append(line.strip().split(','))
                    current_line_listed_format = line.strip().split(',')
                    for variable, key_value in enumerate (current_line_listed_format):
                        if dict_key_initial_count == True:
                            self.variable_keys[self.random_variable[variable]] = [key_value] 
                            if variable == self.no_of_random_variable-1:
                                dict_key_initial_count = False
                        if key_value not in (self.variable_keys[self.random_variable[variable]]):
                            self.variable_keys[self.random_variable[variable]].append(key_value)
                    self.data_instances = self.data_instances + 1    
    
    #estimate probablity for the parameter
    def estimate_parameter_probablity(self):
        for var_index in range(0, (self.no_of_random_variable)):
            Training_data_count = {}
            for Training_values in self.Training_values:
                prob = Training_values[var_index] if var_index is self.no_of_random_variable-1 else Training_values[var_index]+'|'+Training_values[self.no_of_random_variable-1]
                try:
                    Training_data_count[prob] += 1
                except Exception:
                    Training_data_count[prob] = 1
            self.estimated_parameters_prob[self.random_variable[var_index]] =  Training_data_count
            self.check_missing_parameters(var_index, Training_data_count)
    
    #identify missing parameters
    def check_missing_parameters(self, rand_var_index, Training_data_count):
        for key_val in self.variable_keys[self.random_variable[rand_var_index]]:
            for predictor_val in self.variable_keys[self.predictor]:
                if self.random_variable[rand_var_index] == self.predictor:
                    break
                given_variables = key_val+'|'+predictor_val
                if given_variables not in Training_data_count:
                    Training_data_count[given_variables] = self.missing_value
    
    #calculate probablity distribution
    def calculate_probablity(self):
        for rand_var, key_val in self.estimated_parameters_prob.items():
            prob_distribution = {}
            prior_counts = self.estimated_parameters_prob[self.predictor]
            for key, val in key_val.items():
                prior_var = key.split('|')
                prob = float(val/self.data_instances) if (len(prior_var) == 1)  else float(val/prior_counts[prior_var[1]])     
                prob_distribution[key] = prob
            self.probablities[rand_var] = prob_distribution

    #test probability query
    def test_probablity_query(self):
        predictor = {}
        get_pred_key = []
        prob_distribution_false = 1
        prob_distribution_true = 1
        for rand_var, key_val in self.probablities.items():
            prior_counts = self.estimated_parameters_prob[self.predictor]
            for key, val in key_val.items():
                cond_var = key.split('|')
                get_pred_key = list(prior_counts.keys())
                if rand_var == self.predictor:
                    predictor = key_val
                    self.probablity_distribution[get_pred_key[1]] = predictor[get_pred_key[1]] * prob_distribution_true
                    self.probablity_distribution[get_pred_key[0]] = predictor[get_pred_key[0]] * prob_distribution_false
                    break
                for test_rand_var, test_val in self.test_probablity_var_type.items():
                    if cond_var[1] == get_pred_key[0] and cond_var[0] == test_val and test_rand_var == rand_var:
                        prob_distribution_false =  prob_distribution_false * val
                    if cond_var[1] == get_pred_key[1] and cond_var[0] == test_val and test_rand_var == rand_var:
                        prob_distribution_true =  prob_distribution_true * val

        print(self.probablities)
        print("\nUnNormalised Distribution =>", self.probablity_distribution)
        self.get_normalised_distribution(get_pred_key)

    #get normalised distribution
    def get_normalised_distribution(self, get_pred_key):
        prob_true = self.probablity_distribution[get_pred_key[1]]
        prob_false = self.probablity_distribution[get_pred_key[0]]
        self.normalised_distribution[get_pred_key[1]] = prob_true / (prob_true + prob_false)
        self.normalised_distribution[get_pred_key[0]] = prob_false / (prob_true + prob_false)
        self.pred.append(self.normalised_distribution)
        print("Normalised Distribution =>",self.normalised_distribution)
        #comp_performance = Compute_Performance(self.Training_values, self.pred, self.random_variable)
        #comp_performance.compute_performance()

    #take input to calculate probablity
    def input(self):
        print("\nENTER INPUT VECTORS")
        print("Random Varibales are =>", self.random_variable)
        print("Predictor Variable => %s" % self.predictor)
        no_of_random_variable = int(input("Enter no of Evidence variable:"))
        for var in range(0, no_of_random_variable):
            rand_var =  input("Enter Evidence variable %d:" % (var+1))
            if rand_var in self.variable_keys:
                print(self.variable_keys[rand_var])
            type = input("Enter Type of Evidence Variable:")
            self.test_probablity_var_type[rand_var] = type

    #debug values
    def debug(self):
        print(self.random_variable)
        print(tabulate(self.Training_values))
        print(self.variable_keys)
        print(self.Training_values)
        print(self.estimated_parameters_prob)
        print(self.data_instances)

if len(sys.argv) != 3:
    print("USAGE: NB_Classifier.py [train_file.csv] [test_file.csv]")
    exit(0)

#creating NB classifier object
train_data = NB_Classifier()
train_data.read_data()
train_data.estimate_parameter_probablity()
train_data.debug()
train_data.calculate_probablity()
train_data.input()
train_data.test_probablity_query()

