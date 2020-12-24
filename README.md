# Natural-Language-Processing-
we performed Natural Language Processing to classify a news article as fake or non-fake in the light of several outstanding research papers in NLP, ML & DL. We took the dataset from Kaggle.com which has 20,800 rows/data and four input feature vector and one output vector. We took “TEXT” column for our analysis which has content of around 500 words in one row. As the output was already given so we proceeded with some of the supervised machine learning algorithms such as Support Vector Machines using kernel function as RBF(Radial Bessel function) and then Sigmoid function, Multinomial Naive Bayes, Multinomial Naive Bayes with hyper-parameter, Passive Aggressive Classifier, Decision Tree. 
Some pre-processing was done before feeding it to the models such as dropping of the null values as we had only 39 null values in the text column which won’t hamper the efficiency, removing stop words(is, the, are etc) form the text we had as it won’t contribute to the analysis much, stemming of the sentence was done, substitution of vacant space in place of any character other than a-z or A-Z, lastly we lowered the whole sentence so that the machine may not look the same words differently. After doing these steps of text preprocessing a corpus is made in which we kept all the finally processed words. 

We divide the experiments in 4 parts— 

First two experiments were using Discrete Semantics (Bag of words model, TfIdf) for sparse representation of the input feature vector. Afterwards we are using Distributional Semantics which is one of the most successful idea in Statistical NLP (Natural Language processing) for dense representation of the input feature vector

Experiment 1
On this corpus we applied Bag of Words Model with max_featurs of 10,000, n_gram range of (1,3) and the test size of 10%. This will convert the input vector in digits, after tokenising each words, that can easily be understood by machine learning algorithm. Out of all six algorithms logistic regression give the best accuracy(96%) followed by passive aggressive classifier as 95.5%. 

Experiment 2
The same pre-processing procedures were applied for making the corpus with max_feature of 20,000 and n_gram range of (1,4) and the test size of 10%. There was an increase in accuracy for Naive Bayes, NB with hyper-parmater and in passive aggressive classifier but no change in case of logistic regression whereas we saw a decrease in efficiency in case of SVM with RBF and SVM with Sigmoid. 
