
#Some method taken from open source
#least solution done by Manigandan Sivalingam

import csv 
import math
import time
import numpy as np
import pandas as pd

#Library used from pgmpy
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from sklearn import metrics
from pgmpy.metrics import log_likelihood_score
from pgmpy.metrics import structure_score
from pgmpy.estimators import HillClimbSearch

#taken from pgmpy documentation
heartDisease = pd.read_csv('heart-data-discretized-train.csv')
heartDisease = heartDisease.replace('?',np.nan)
file = 'heart-data-discretized-test.csv'
print('Sample instances from the dataset are given below')
print(heartDisease.head())

print('\n Attributes and datatypes')
print(heartDisease.dtypes)

#taken from pgmpy documentation
#model input given after conditional independence from RemovingEdgesPC.py
#where RemovingEdgesPC.py will remove unwantes edges from a model
model1= BayesianModel([('age','target'),('sex','target') , ('chol','target'), ('restecg', 'target'),('thalach','target'),('thal','thalach','target'),('fbs','thal'),('slope','thalach','target','chol', 'restecg'),('oldpeak','thalach','slope'),('age','fbs','target','thal','slope','restecg'),('sex','slope','chol','restecg'),('cp','target','slope','restecg'), ('ca','thalach','target','fbs','age','oldpeak'),('exang','thalach','thal','target','fbs','age','restecg','sex'),('trestbps','thalach','target','age','chol','thal','restecg','oldpeak', 'cp')])
model2= BayesianModel([('age','target'),('sex','target') ,('trestbps', 'target'), ('chol','target'), ('restecg', 'target'),('thalach','target'),('thal','target'),('fbs','thal'),('slope','thalach','target','chol', 'restecg'),('oldpeak','thalach','slope'),('age','target','thal','slope','restecg'),('sex','slope','chol','restecg'),('cp','target','slope','restecg'), ('ca','thalach','target','fbs','age','oldpeak'),('exang','thalach','thal','target','fbs','age','restecg','sex'),('thalach','target','age','chol','thal','restecg','oldpeak', 'cp')])
model3= BayesianModel([('age','target', 'sex'),('sex','target','trestbps') , ('chol','target'), ('sex','chol'),('fbs','target'),('restecg', 'target'),('thalach','target'),('thal','thalach','target', 'age', 'sex'),('fbs','thal'),('slope','thalach','target','chol', 'restecg'),('oldpeak','thalach','slope'),('age','slope','restecg'),('sex','chol','restecg'),('cp','target','slope'), ('ca','thalach','target','fbs','age','oldpeak'),('exang','thalach','thal','target','restecg','sex'),('trestbps','thalach','age','restecg','oldpeak', 'cp')])
model4= BayesianModel([('age','sex'),('age','target'),('age','fbs'),('age','thal'),('age','trestbps'),('age','ca'),('age','exang'),('age','slope'),('age','restecg'),('sex','exang') , ('sex','slope'), ('sex','chol'),('sex','restecg'), ('cp','trestbps'), ('chol','target'), ('restecg', 'target'),('thalach','target'),('thal','thalach','target'),('fbs','thal'),('slope','thalach','target','chol', 'restecg'),('oldpeak','thalach','slope'),('age','fbs','target','thal','slope','restecg'),('sex','slope','chol','restecg'),('cp','target','slope','restecg'), ('ca','thalach','target','fbs','age','oldpeak'),('exang','thalach','thal','target','fbs','age','restecg','sex'),('trestbps','thalach','target','age','chol','thal','restecg','oldpeak', 'cp')])
model5= BayesianModel([('age','sex'),('age','target'),('age','fbs'),('age','thal'),('age','trestbps'),('age','ca'),('age','exang'),('age','slope'),('age','restecg'),('sex','exang') , ('sex','slope'), ('sex','chol'),('sex','restecg'), ('cp','trestbps'), ('cp','target'), ('cp','thal'),('cp','restecg'),('trestbps','thalach'),('trestbps','target'),('trestbps','chol'),('trestbps','thal'),('trestbps','restecg'),('trestbps','oldpeak'),('chol','target'), ('restecg', 'target'),('thalach','target'),('thal','thalach','target'),('fbs','thal'),('slope','thalach','target','chol', 'restecg'),('oldpeak','thalach','slope'),('age','fbs','target','thal','slope','restecg'),('sex','slope','chol','restecg'),('cp','target','slope','restecg'), ('ca','thalach','target','fbs','age','oldpeak'),('exang','thalach','thal','target','fbs','age','restecg','sex'),('trestbps','thalach','target','age','chol','thal','restecg','oldpeak', 'cp')])
model6= BayesianModel([('age','sex'),('age','target'),('age','fbs'),('age','thal'),('age','trestbps'),('age','ca'),('age','exang'),('age','slope'),('age','restecg'),('sex','exang') , ('sex','slope'), ('sex','chol'),('sex','restecg'), ('cp','trestbps'), ('cp','target'), ('cp','thal'),('cp','restecg'),('trestbps','thalach'),('trestbps','target'),('trestbps','chol'),('trestbps','thal'),('trestbps','restecg'),('trestbps','oldpeak'),('chol','target'), ('chol','slope'), ('chol','restecg'), ('fbs','exang'), ('fbs','ca'), ('fbs','thal'), ('restecg', 'target'),('restecg', 'slope'),('restecg', 'exang'),('thalach','target'),('thal','thalach','target'),('fbs','thal'),('slope','thalach','target','chol', 'restecg'),('oldpeak','thalach','slope'),('age','fbs','target','thal','slope','restecg'),('sex','slope','chol','restecg'),('cp','target','slope','restecg'), ('ca','thalach','target','fbs','age','oldpeak'),('exang','thalach','thal','target','fbs','age','restecg','sex'),('trestbps','thalach','target','age','chol','thal','restecg','oldpeak', 'cp')])
model7= BayesianModel([('age','target'),('sex','target') , ('chol','target'), ('restecg', 'target'),('thalach','target'),('thal','thalach','target'),('fbs','thal'),('slope','thalach','target','chol', 'restecg'),('oldpeak','thalach','slope'),('age','fbs','target','thal','slope','restecg'),('sex','slope','chol','restecg'),('cp','target','slope','restecg'), ('ca','thalach','target','fbs','age','oldpeak'),('exang','thalach','thal','target','fbs','age','restecg','sex'),('trestbps','thalach','target','age','chol','thal','restecg','oldpeak', 'cp')])
model8= BayesianModel([('age','sex'),('age','target'),('age','fbs'),('age','thal'),('age','trestbps'),('age','ca'),('age','exang'),('age','slope'),('age','restecg'),('sex','exang') , ('sex','slope'), ('sex','chol'),('sex','restecg'), ('cp','trestbps'), ('cp','target'), ('cp','thal'),('cp','restecg'),('trestbps','thalach'),('trestbps','target'),('trestbps','chol'),('trestbps','thal'),('trestbps','restecg'),('trestbps','oldpeak'),('chol','target'), ('chol','slope'), ('chol','restecg'), ('fbs','exang'), ('fbs','ca'), ('fbs','thal'), ('restecg', 'target'),('restecg', 'slope'),('restecg', 'exang'),('thalach','target'),('thalach','slope'),('thalach','ca'),('thalach','thal'),('thalach','exang'),('thalach','oldpeak'), ('exang','thal'),('exang','target'),('fbs','thal'),('slope','target','chol', 'restecg'),('age','fbs','target','thal','slope','restecg'),('sex','slope','chol','restecg'),('cp','target','slope','restecg'), ('ca','target','fbs','age','oldpeak'),('exang','thal','target','fbs','age','restecg','sex'),('trestbps','target','age','chol','thal','restecg','oldpeak', 'cp')])

#taken from pgmpy documentation
print('\nLearning CPD using Maximum likelihood estimators')
model.fit(heartDisease,estimator=MaximumLikelihoodEstimator)
data = model.simulate(int(1e4))
print('\n Inferencing with Bayesian Network:')
HeartDiseasetest_infer = VariableElimination(model)
print('\n 1. Probability of HeartDisease given evidence= restecg')

#list of models
modelsList = [model1, model2, model3, model4, model5, model6]

#pgmpy documentaion
ll = log_likelihood_score(model, data)
ScoreG = structure_score(model, data, scoring_method="bic")

#Author Manigandan
MaxScore = ScoreG
BicScoreList = []
Log_LikelihoodList = []
print(MaxScore)
count = 0
for models in modelsList:
    models.fit(heartDisease,estimator=MaximumLikelihoodEstimator)
    data = model.simulate(int(1e4))
    ll = log_likelihood_score(models, data)
    ScoreG = structure_score(models, data, scoring_method="bic")
    BicScoreList.append(math.trunc(ScoreG))
    count = count+1
    print("count = ", count+1)
    print(ScoreG)
    if MaxScore > ScoreG:
        MaxScore = ScoreG
        model = models
        print("Max Score=", MaxScore)
        print(count)

#pgmy documentaion
HeartDiseasetest_infer = VariableElimination(model)
mm = model.to_markov_model()
print("\n\n\n\n\nEdges")
print(mm.edges())
#calculate inference time
inference_time = time.time()
q1 = HeartDiseasetest_infer.query(variables=['target'],evidence={'sex':0, 'cp':3})
inference_time = time.time() - inference_time
print(inference_time)
print("Probablity",q1)


#compute performance author: herberito
count = 0
Y_true = []
Y_pred = []
Y_prob = []
with open(file, 'r') as csvfile:
    #modified changes by Author: Manigandan
    datareader = csv.reader(csvfile)
    for row in datareader:
        if count == 0:
            count = 1
            continue
        q1 = HeartDiseasetest_infer.query(variables=['target'],evidence={'age':int(row[0]),'sex':int(row[1]), 'cp':int(row[2]), 'trestbps':int(row[3]),
        'chol':int(row[4]), 'fbs':int(row[5]),  'restecg':int(row[6]) , 'thalach':int(row[7]),  'exang':int(row[8]) ,'oldpeak':int(row[9]), 'slope':int(row[10]),
        'ca':int(row[11]) , 'thal':int(row[12])})
        target_value = row[13]

        #workshop material to compute performance
        if target_value == 'yes': Y_true.append(1)
        elif target_value == 'no': Y_true.append(0)
        elif target_value == '1': Y_true.append(1)
        elif target_value == '0': Y_true.append(0)
        
        pred = list(q1.values)
        pred_max = max(pred)
        Y_prob.append(pred[int(row[13])])
        best_key = str(pred.index(pred_max))

        if best_key == 'yes': Y_pred.append(1)
        elif best_key == 'no': Y_pred.append(0)
        elif best_key == '1': Y_pred.append(1)
        elif best_key == '0': Y_pred.append(0)


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

#calculate loglikelihood bicscore for a model
ll = log_likelihood_score(model, data)
structScore = structure_score(model, data, scoring_method="bic")

print("log Likelihood = ", ll)
print("Structure Score = ", structScore)

print("Models         | Model1   Model2  Model3  Model4  Model5  Model6")
print("Log-Likelihood |",Log_LikelihoodList[0],Log_LikelihoodList[1],Log_LikelihoodList[2],Log_LikelihoodList[3],Log_LikelihoodList[4],Log_LikelihoodList[5])
print("BIC Score      |",BicScoreList[0], BicScoreList[1], BicScoreList[2], BicScoreList[3], BicScoreList[4], BicScoreList[5])
