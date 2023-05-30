#Author: Manigandan Sivalingam
#Likelihood weighting
#Approximate Inference Method
#two function called from workshop material

import sys
import random
import BayesNetUtil as bnu
from BayesNetReader import BayesNetReader
from NB_Classifier_metrics import *
from BayesNetApproxInference1 import BayesNetApproxInference

#Likelihood Weighting
class Likelihood_Weighting():
    #no of loglikelihood samples
    Likelihood_samples = 0
    Likelihood_prediction = {}
    inference_time = 0
    rand_vars = []
    rv_all_values = []
    
    #Author: Manigandan Sivalingam
    #Likelihood weighting
    #Approximate Inference Method
    def Likelihood_weighting(self, evidence):
        
        sum = 0
        weights = [0,0]
        #probablity distribution for likelihood
        likelihood_probablity_distribution = {}
        query_variable = self.query["query_var"]
        print(query_variable)
        evidence = self.query["evidence"]
        test = 1
        if test == 1:
            evidence = self.query["evidence"]
        C = {}
        
        query_key = '1'

        for value in self.bn["rv_key_values"][query_variable]:
            value = value.split("|")[0]
            C[value] = 0
        print("\nLIKELIHOOD SAMPLING")
        #calculate inference time
        self.inference_time = time.time()
        for i in range(0, self.num_samples):
            X = self.prior_sample()
            sample = X
            #calculate weight for evidence variable
            result, weight_of_s = self.weighted_sample(X, evidence)
            
            #summing the weights
            if (result == True):
                if (sample[query_variable] == query_key):
                    weights[0] = weights[0] + weight_of_s
                    self.Likelihood_samples = self.Likelihood_samples + 1
                else:
                    weights[1] = weights[1] + weight_of_s
                    self.Likelihood_samples = self.Likelihood_samples + 1

        sum = (weights[0])/(weights[0] + weights[1])
        #normalizing      
        likelihood_probablity_distribution["0"] = sum
        likelihood_probablity_distribution["1"] = 1-sum
        self.inference_time = time.time() - self.inference_time
        print("Inference Time", self.inference_time)
        print("Number of Likelihood Samples", self.Likelihood_samples)
        print("Likelihood Probablity Distribution", likelihood_probablity_distribution)
        return (likelihood_probablity_distribution)

    def weighted_sample(self, X, evidence):
        weight = 1
        p= 0
        count = 0
        #based on evidence calculate weight
        for variable, value in evidence.items():
            if X[variable] == value:
                p = bnu.get_probability_given_parents(variable, value, X, self.bn)
                weight = weight * p
                count = count+1
                if(count == len(evidence.items())):
                    print("%s => %f", X, weight)
                    self.Likelihood_samples = self.Likelihood_samples + 1
                    return True, weight 
        return False, weight

    #test the likelihood weighting test file to caluculate metrics
    def test_learn_prob(self):
        evidence = []
        pred_prob = {}
        count = 0
        for i in self.rv_all_values:
            evidence = {}
            for j in range(0, 4):
                if (j == 0) or (j == 2):
                    evidence[self.rand_vars[j]] = i[j] 
                    #print(evidence)
            pred = self.Likelihood_weighting(evidence)
            pred_prob[count] = pred
            print(pred_prob[count])
            if count == 20:
                break
            count = count+1
        return pred_prob