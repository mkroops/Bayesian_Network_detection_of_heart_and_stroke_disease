#############################################################################
# BayesNetExactInference.py
#
# This program implements the algorithm "Inference by Enumeration", which
# makes use of BayesNetsReader to facilitate reading data of a Bayes net via
# the object self.bn created by the inherited class (BayesNetReader). It also
# makes use of miscellaneous methods implemented in BayesNetUtil.
# Its purpose is to answer probabilistic queries such as P(Y|X=true,Z=false).
# This implementation is agnostic of the data and provides a general
# implementation that can ne used across datasets by providing a config file.
#
# WARNING: This code has not been thoroughly tested.
#
# Version: 1.0, Date: 06 October 2022, first version
# Version: 1.2, Date: 21 October 2022, revised version (more query compatible)
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import sys
import BayesNetUtil as bnu
from BayesNetReader import BayesNetReader
import numpy as np
import pandas as pd
import csv 
import time
from sklearn import metrics


class BayesNeExactInference(BayesNetReader):
    query = {}
    prob_dist = {}
    variables = {}
    evidence = {}
    inference_time = 0

    def __init__(self):
        if len(sys.argv) != 3:
            print("USAGE: BayesNetInference.py [your_config_file.txt] [query]")
            print("EXAMPLE> BayesNetInference.py config-alarm.txt \"P(B|J=true,M=true)\"")
        else:
            file_name = sys.argv[1]
            prob_query = sys.argv[2]
            super().__init__(file_name)
            self.query = bnu.tokenise_query(prob_query)
            #calculate inference time modified by manigandan
            self.inference_time = time.time()
            self.prob_dist = self.enumeration_ask()
            self.inference_time = time.time() - self.inference_time
            print("Inference Time", self.inference_time)
            normalised_dist = bnu.normalise(self.prob_dist)
            #compute performance slightly modified by modified by manigandan
            self.compute_performance()
            print("unnormalised probability_distribution="+str(self.prob_dist))
            print("normalised probability_distribution="+str(normalised_dist))

    def enumeration_ask(self):
        #print("\nSTARTING Inference by Enumeration...")
        Q = {}
        for value in self.bn["rv_key_values"][self.query["query_var"]]:
            value = value.split('|')[0]
            Q[value] = 0

        for value, probability in Q.items():
            value = value.split('|')[0]
            self.variables = self.bn["random_variables"].copy()
            self.evidence = self.query["evidence"].copy()
            self.evidence[self.query["query_var"]] = value
            #print("value",value)
            probability = self.enumerate_all(self.variables, self.evidence)
            Q[value] = probability
            print("\tQ="+str(Q))
        #print("End")
        return Q

    def enumerate_all(self, variables, evidence):
        #print("\nCALL to enumerate_all(): V=%s E=%s" % (variables, evidence))
        if len(variables) == 0:
            return 1.0

        V = variables[0]

        if V in evidence:
            v = evidence[V].split('|')[0]
            p = bnu.get_probability_given_parents(V, v, evidence, self.bn)
            variables.pop(0)
            return p*self.enumerate_all(variables, evidence)

        else:
            sum = 0
            evidence_copy = evidence.copy()
            for v in bnu.get_domain_values(V, self.bn):
                evidence[V] = v
                p = bnu.get_probability_given_parents(V, v, evidence, self.bn)
                rest_variables = variables.copy()
                rest_variables.pop(0)
                sum += p*self.enumerate_all(rest_variables, evidence)
                evidence = evidence_copy

            return sum

     #compute performance slightly modified by modified by manigandan
    def compute_performance(self):
        file = 'heart-data-discretized-test.csv'
        count = 0
        Y_true = []
        Y_pred = []
        Y_prob = []
        with open(file, 'r') as csvfile:
            datareader = csv.reader(csvfile)
            for row in datareader:
                if count == 0:
                    count = count + 1
                    continue
                
                #stroke dataset
                #self.evidence={'gender':(row[0]),'age':(row[1]), 'hypertension':(row[2])}
                #,'heart_disease':(row[3]),
                #'ever_married':(row[4]), 'work_type':(row[5]),  'Residence_type':(row[6]) , 'avg_glucose_level':(row[7]),  'bmi':(row[8]) ,
                #'smoking_status':(row[9])}
                #for heart data set
                self.evidence={'age':int(row[0]),'sex':int(row[1]), 'cp':int(row[2]), 'trestbps':int(row[3]),
                'chol':int(row[4]), 'fbs':int(row[5]),  'restecg':int(row[6]) , 'thalach':int(row[7]),  'exang':int(row[8]) ,'oldpeak':int(row[9]), 'slope':int(row[10]),
                'ca':int(row[11]) , 'thal':int(row[12])}

                p = self.enumeration_ask()
                predicted_prob = bnu.normalise(p)
                print(predicted_prob)
                q1=[predicted_prob['0'], predicted_prob['1']]
                #print(q1)
                target_value = row[13]
                #print("target", target_value)
                if target_value == 'yes': Y_true.append(1)
                elif target_value == 'no': Y_true.append(0)
                elif target_value == '1': Y_true.append(1)
                elif target_value == '0': Y_true.append(0)
                pred = q1
                pred_max = max(pred)
                Y_prob.append(pred[int(row[13])])
                best_key = str(pred.index(pred_max))

                if best_key == 'yes': Y_pred.append(1)
                elif best_key == 'no': Y_pred.append(0)
                elif best_key == '1': Y_pred.append(1)
                elif best_key == '0': Y_pred.append(0)
                print("Count", count)
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
            

BayesNeExactInference()
