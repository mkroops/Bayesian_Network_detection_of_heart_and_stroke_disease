# Bayesian_Network_detection_of_heart_and_stroke_disease

I. DISCRETE BAYESIAN NETWORK
A. Naive Bayes Classifier
The Naive Bayes Classifier algorithm is the simplest classi-
fication algorithm which is based on Bayes rules and assumes
all the features are conditional independent to each other.
Naive Bayes reduces the number of parameters by making a
conditional independent assumption. Methodology used to
solve this probabilistic query is Naive Bayes
B. What are the difficulties faced in the Discrete Bayesian
network?
Bayesian network is richer in structure but when the number
of random variables increases it takes lots of computational
time to generate Conditional probability table (CPT) for the
predefined structure or randomly generated structure. Finding
a good structure will require iteration on multiple hypothetical
structure to test multiple possibilities. For answering simple
probabilistic queries Naive Bayes classifier suits well  but
accuracy lacks since the features are conditionally indepen-
dent.
C. Analysis of Naive Bayes
Method used for estimating the parameters is maximum
likelihood estimation. some values are missing while estimat-
ing the parameters so zero probability are noticed, so it is
avoided by Laplace smoothing . If Laplace smoothing is not
done, Answer for the probabilistic query will be inaccurate.
Laplace smoothing for one variable is given by
P (X = x) = count(x) + 1
count(X) + |X| (1)
where X is random variable and |X| is domain size of variable
X. Similarly Laplace smoothing for two variable is given by
P (x|y) = count(x|y) + 1
count(y) + |X| (2)
where (x|y) is the conditional probability and y is random
variable. Unnormalized distribution will be noticed in which
the sum of ‘0’ and ‘1’ is not equal to 1 so convert that into
normalized distribution.
D. Comparison of Naive bayes classifier and discrete
bayesian network prdefined structure based on probabilistic
query of Stroke Dataset.
Algorithm 0 1
Naive Bayes
classifier
0.9604 0.0309
Bayesian Net
Exact
0.9595 0.0404
.
Table 1.Statistics of two algorithms
Probabilistic answers for Naive Bayes and Bayesian net
predefined structure vary slightly in Table.1, it is because
Bayesian Network predefined structure is more dependent on
structure so it gives more accurate answer [2].
E. Training time taken for Naive Bayes classifier and Bayesian
network predefined structure.
Time Taken Naive Bayes Bayesian Net
Training(s) 0.0356 s > 3600 s .
Table 2.Training time taken for two algorithms
It is noticed in Table.2 Bayesian Net predefined structure
takes more computational time because when random variable
increases, dependency on structure gets complicated to train
the data set.
F. Output of Naive Bayes class for heart data set.
Probabilistic
query
0 1
P(target | sex
=0, cp = 3)
0.1567 0.8432
Table.3 Probabilistic query for Heart
G. Output of Naive Bayes for stroke data set
Probabilistic
query
0 1
P(stroke |
gender =
Male, age = 2,
smok-
ing status=
Formely
smoked))
0.9604 0.0309
.
Table.4 Probabilistic query for Stroke
