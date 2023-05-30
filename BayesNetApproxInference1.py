#############################################################################
# BayesNetApproxInference.py
#
# This program implements the algorithm "Rejection Sampling", which
# imports functionalities to facilitate reading data of a Bayes net via
# the object self.bn created by the inherited class BayesNetReader.
# Its purpose is to answer probabilistic queries such as P(Y|X=true,Z=false).
# This implementation is agnostic of the data and provides a general
# implementation that can ne used across datasets by providing a config file.
#
#############################################################################

import sys
import random
import BayesNetUtil as bnu
from BayesNetReader import BayesNetReader
from NB_Classifier_metrics import *
#from Likelihood_Weighting import Likelihood_Weighting

class BayesNetApproxInference(BayesNetReader):
    query = {}
    prob_dist = {}
    seeds = {}
    num_samples = None
    Likelihood_samples = 0
    Likelihood_prediction = {}
    inference_time = 0
    
    rand_vars = []
    rv_all_values = []

    def __init__(self):
        if len(sys.argv) != 4:
            print("USAGE> BayesNetApproxInference.py [your_config_file.txt] [query] [num_samples]")
            print("EXAMPLE> BayesNetApproxInference.py config-alarm.txt \"P(B|J=true,M=true)\" 10000")
        else:
            #file_name_test = "stroke-data-discretized-test.csv"
            #self.read_data(file_name_test)
            file_name = sys.argv[1]
            prob_query = sys.argv[2]
            self.num_samples = int(sys.argv[3])
            super().__init__(file_name)
            self.query = bnu.tokenise_query(prob_query)
            Nb = NB_Classifier(None)
            self.rand_vars, self.rv_all_values = Nb.read_data("heart-data-discretized-test.csv")
            #LW = Likelihood_weighting()
            #LW.Likelihood_weighting(None)
            self.Likelihood_weighting(None)
            #pred = self.test_learn_prob()
            #self.compute_performance(pred)
            self.prob_dist = self.rejection_sampling()
            print("Rejection Probability_distribution="+str(self.prob_dist))

    def Likelihood_weighting(self, evidence):
        
        sum = 0
        weights = [0,0]
        likelihood_probablity_distribution = {}
        query_variable = self.query["query_var"]
        print(query_variable)
        evidence = self.query["evidence"]
        print("evidence", evidence)
        test = 1
        if test == 1:
            evidence = self.query["evidence"]
        #evidence = self.query["evidence"]
        C = {}
        
        query_key = '1'
        '''if "=" in query_variable:
            key_sep = query_variable.split("=")
            query_variable = key_sep[0]
            query_key = key_sep[1]'''

        #print(self.bn["rv_key_values"])
        for value in self.bn["rv_key_values"][query_variable]:
            value = value.split("|")[0]
            C[value] = 0
        print("\nLIKELIHOOD SAMPLING")
        self.inference_time = time.time()
        for i in range(0, self.num_samples):
            X = self.prior_sample()
            sample = X
            result, weight_of_s = self.weighted_sample(X, evidence)
            
            if (result == True):
                if (sample[query_variable] == query_key):
                    weights[0] = weights[0] + weight_of_s
                    self.Likelihood_samples = self.Likelihood_samples + 1
                else:
                    weights[1] = weights[1] + weight_of_s
                    self.Likelihood_samples = self.Likelihood_samples + 1

        sum = (weights[0])/(weights[0] + weights[1])       
        #print("weight0, weight1, total weight", weights[0], weights[1],(weights[0] + weights[1]))
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
        #print(evidence)
        for variable, value in evidence.items():
            if X[variable] == value:
                p = bnu.get_probability_given_parents(variable, value, X, self.bn)
                weight = weight * p
                count = count+1
                #print(value)
                if(count == len(evidence.items())):
                    print("%s => %f", X, weight)
                    self.Likelihood_samples = self.Likelihood_samples + 1
                    return True, weight 
        return False, weight


    def rejection_sampling(self):
        print("\nREJECTION SAMPLING")
        query_variable = self.query["query_var"]
        evidence = self.query["evidence"]
        C = {}

        # initialise vector of counts
        for value in self.bn["rv_key_values"][query_variable]:
            value = value.split("|")[0]
            C[value] = 0

        # loop to increase counts when the sampled vector consistent w/evidence
        self.inference_time = 0
        self.inference_time = time.time()
        for i in range(0, self.num_samples):
            X = self.prior_sample()
            if self.is_compatible_with_evidence(X, evidence):
                value_to_increase = X[query_variable]
                C[value_to_increase] += 1
        self.inference_time = time.time() - self.inference_time
        print("Inference Time", self.inference_time)
        print("No of Rejection samples", C[value_to_increase])
        return bnu.normalise(C)

    def prior_sample(self):
        X = {}
        sampled_var_values = {}
        for variable in self.bn["random_variables"]:
            X[variable] = self.get_sampled_value(variable, sampled_var_values)
            sampled_var_values[variable] = X[variable]

        return X

    def get_sampled_value(self, V, sampled):
        # get the conditional probability distribution (cpt) of variable V
        parents = bnu.get_parents(V, self.bn)
        cpt = {}
        prob_mass = 0

        # generate a cumulative distribution for random variable V
        if parents is None:
            for value, probability in self.bn["CPT("+V+")"].items():
                prob_mass += probability
                cpt[value] = prob_mass

        else:
            for v in bnu.get_domain_values(V, self.bn):
                p = bnu.get_probability_given_parents(V, v, sampled, self.bn)
                prob_mass += p
                cpt[v] = prob_mass

        # check that the cpt sums to 1 (or almost)
        if prob_mass < 0.999 and prob_mass > 1:
            print("ERROR: CPT=%s does not sum to 1" % (cpt))
            exit(0)

        return self.sampling_from_cumulative_distribution(cpt)

    def sampling_from_cumulative_distribution(self, cumulative):
        random_number = random.random()
        for value, probability in cumulative.items():
            if random_number <= probability:
                random_number = random.random()
                return value.split("|")[0]

        print("ERROR couldn't do sampling from:")
        print("cumulative_dist="+str(cumulative))
        exit(0)

    def is_compatible_with_evidence(self, X, evidence):
        for variable, value in evidence.items():
            if X[variable] != value:
                return False
        return True

    
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

    def compute_performance(self, predicted_output):
        
        Y_true = []
        Y_pred = []
        Y_prob = []
        count = 0

        #print(predicted_output)
        # obtain vectors of categorical and probabilistic predictions
        
        for i in self.rv_all_values:
            target_value = i[len(self.rand_vars)-1]
            if target_value == 'yes': Y_true.append(1)
            elif target_value == 'no': Y_true.append(0)
            elif target_value == '1': Y_true.append(1)
            elif target_value == '0': Y_true.append(0)

            #predicted_output = self.predictions[i][target_value]
            #print(predicted_output)
            pred = predicted_output[count][target_value]
            Y_prob.append(pred)
            best_key = max(predicted_output[count] , key=predicted_output[count].get)
            if best_key == 'yes': Y_pred.append(1)
            elif best_key == 'no': Y_pred.append(0)
            elif best_key == '1': Y_pred.append(1)
            elif best_key == '0': Y_pred.append(0)
            if count == 20:
                break
            count = count + 1

        P = np.asarray(Y_true)+0.00001 # constant to avoid NAN in KL divergence
        Q = np.asarray(Y_prob)+0.00001 # constant to avoid NAN in KL divergence

        # calculate metrics: accuracy, auc, brief, kl, training/inference times
        acc = metrics.balanced_accuracy_score(Y_true, Y_pred)
        fpr, tpr, _ = metrics.roc_curve(Y_true, Y_prob, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        brier = metrics.brier_score_loss(Y_true, Y_prob)
        kl_div = np.sum(P*np.log(P/Q))
        print("PERFORMANCE:")
        print("Balanced Accuracy="+str(acc))
        print("Area Under Curve="+str(auc))
        print("Brier Score="+str(brier))
        print("KL Divergence="+str(kl_div))
        #print("Training Time="+str(self.training_time)+" secs.")
        #print("Inference Time="+str(self.inference_time)+" secs.")

BayesNetApproxInference()
