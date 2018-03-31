"""
Author      : Yi-Chieh Wu, Sriram Sankararman
Description : Twitter
"""

from string import punctuation

import numpy as np

# !!! MAKE SURE TO USE SVC.decision_function(X), NOT SVC.predict(X) !!!
# (this makes ``continuous-valued'' predictions)
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics

import collections

######################################################################
# functions -- input/output
######################################################################

def read_vector_file(fname):
    """
    Reads and returns a vector from a file.
    
    Parameters
    --------------------
        fname  -- string, filename
        
    Returns
    --------------------
        labels -- numpy array of shape (n,)
                    n is the number of non-blank lines in the text file
    """
    return np.genfromtxt(fname)


def write_label_answer(vec, outfile):
    """
    Writes your label vector to the given file.
    
    Parameters
    --------------------
        vec     -- numpy array of shape (n,) or (n,1), predicted scores
        outfile -- string, output filename
    """
    
    # for this project, you should predict 70 labels
    if(vec.shape[0] != 70):
        print("Error - output vector should have 70 rows.")
        print("Aborting write.")
        return
    
    np.savetxt(outfile, vec)    


######################################################################
# functions -- feature extraction
######################################################################

def extract_words(input_string):
    """
    Processes the input_string, separating it into "words" based on the presence
    of spaces, and separating punctuation marks into their own words.
    
    Parameters
    --------------------
        input_string -- string of characters
    
    Returns
    --------------------
        words        -- list of lowercase "words"
    """
    
    for c in punctuation :
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def extract_dictionary(infile):
    """
    Given a filename, reads the text file and builds a dictionary of unique
    words/punctuations.
    
    Parameters
    --------------------
        infile    -- string, filename
    
    Returns
    --------------------
        word_list -- dictionary, (key, value) pairs are (word, index)
    """
    
    word_list = {}
    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1a: process each line to populate word_list
        count = 0
        for line in fid:
        	sentence = set(extract_words(line))
        	for word in sentence:
        		# If the word is not in the dictionary
        		if not word_list.has_key(word):
        			word_list[word] = count
        			count += 1
        ### ========== TODO : END ========== ###
    return word_list


def extract_feature_vectors(infile, word_list):
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.
    
    Parameters
    --------------------
        infile         -- string, filename
        word_list      -- dictionary, (key, value) pairs are (word, index)
    
    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of non-blank lines in the text file
                            d is the number of unique words in the text file
    """
    
    num_lines = sum(1 for line in open(infile,'rU'))
    num_words = len(word_list)

    # Creates numpy array with dimensions num_lines x num_words
    feature_matrix = np.zeros((num_lines, num_words))
    word_list = extract_dictionary(infile)

    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1b: process each line to populate feature_matrix
        line_count = 0
        for line in fid:
        	sentence = set(extract_words(line))
        	for word in sentence:
        		# Find the index value and set our feature vector
        		if word_list.has_key(word):
        			index = word_list[word]
        			feature_matrix[line_count][index] = 1
        	line_count += 1
        ### ========== TODO : END ========== ###
        
    return feature_matrix


######################################################################
# functions -- evaluation
######################################################################

def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric based on the agreement between the 
    true labels and the predicted labels.
    
    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1_score', 'auroc', 'precision',
                           'sensitivity', 'specificity'        
    
    Returns
    --------------------
        score  -- float, performance score
    """
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)
    y_label[y_label==0] = 1
    
    ### ========== TODO : START ========== ###
    all_metrics = ['accuracy', 'f1_score', 'auroc', 'precision', 'sensitivity', 'specificity']
    assert(metric in all_metrics), "Unknown metric '{}'".format(metric)

    # part 2a: compute classifier performance
    if metric == 'accuracy':
    	return metrics.accuracy_score(y_true, y_label)
    elif metric == 'f1_score':
    	return metrics.f1_score(y_true, y_label)
    elif metric == 'auroc':
    	return metrics.roc_auc_score(y_true, y_label)
    elif metric == 'precision':
    	return metrics.precision_score(y_true, y_label)
    # sensitivity =  true positive rate = true positives / all positives
    elif metric == 'sensitivity':
    	cm = metrics.confusion_matrix(y_true, y_label)
    	return cm[1][1] / float(cm[1][1] + cm[1][0])
    # specificity = true negatives / all negatives
    else:
    	cm = metrics.confusion_matrix(y_true, y_label)
    	return cm[0][0] / float(cm[0][0] + cm[0][1])
    ### ========== TODO : END ========== ###


def cv_performance(clf, X, y, kf, metric="accuracy"):
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.
    
    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """
    
    ### ========== TODO : START ========== ###
    # part 2b: compute average cross-validation performance
    perf = []

    # This will go through each fold in the CV
    # calculate their performances
    for train_index, test_index in kf:
    	X_train, X_test = X[train_index], X[test_index]
    	y_train, y_test = y[train_index], y[test_index]
    	clf.fit(X_train, y_train)
    	y_pred = clf.decision_function(X_test)
    	perf.append(performance(y_test, y_pred, metric))

    return np.mean(np.array(perf))
    ### ========== TODO : END ========== ###


def select_param_linear(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameter that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        C -- float, optimal parameter value for linear-kernel SVM
        c_perf -- a dictionary containing the the tuples (c, performance)
    """
    
    print 'Linear SVM Hyperparameter Selection based on ' + str(metric) + ':'
    C_range = 10.0 ** np.arange(-3, 3)
    
    ### ========== TODO : START ========== ###
    # part 2c: select optimal hyperparameter using cross-validation
    c_perf = {}

    for c in C_range:
    	svm_clf = SVC(C=c, kernel='linear')
    	avg_perf = cv_performance(clf=svm_clf, X=X, y=y, kf=kf, metric=metric)
    	c_perf[c] = avg_perf

    # Get the optimal C based on the average performance
    return max(c_perf, key=c_perf.get), c_perf
    ### ========== TODO : END ========== ###


def select_param_rbf(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameters of an RBF-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameters that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric  -- string, option used to select performance measure
    
    Returns
    --------------------
        C, gamma -- optimal parameter values for an RBF-kernel SVM, tuple of floats
        score -- the best score for the optimal parameter values C, gamma
    """
    
    print 'RBF SVM Hyperparameter Selection based on ' + str(metric) + ':'
    
    ### ========== TODO : START ========== ###
    # part 3b: create grid, then select optimal hyperparameters using cross-validation
    C_range = 10.0 ** np.arange(-3, 3)
    G_range = 10.0 ** np.arange(-3, 3)

    c_perf = {}

    for c in C_range:
    	for g in G_range:
    		svm_clf = SVC(C=c, kernel='rbf', gamma=g)
    		avg_perf = cv_performance(clf=svm_clf, X=X, y=y, kf=kf, metric=metric)
    		c_perf[(c, g)] = avg_perf

    return max(c_perf, key=c_perf.get), max(c_perf.values())
    # return max(c_perf, key=c_perf.get), c_perf
    ### ========== TODO : END ========== ###


def performance_test(clf, X, y, metric="accuracy"):
    """
    Estimates the performance of the classifier using the 95% CI.
    
    Parameters
    --------------------
        clf          -- classifier (instance of SVC)
                          [already fit to data]
        X            -- numpy array of shape (n,d), feature vectors of test set
                          n = number of examples
                          d = number of features
        y            -- numpy array of shape (n,), binary labels {1,-1} of test set
        metric       -- string, option used to select performance measure
    
    Returns
    --------------------
        score        -- float, classifier performance
    """

    ### ========== TODO : START ========== ###
    # part 4b: return performance on test data by first computing predictions and then calling performance
    # performance(y_true, y_pred, metric="accuracy")

    y_pred = clf.decision_function(X=X)
    score = performance(y_true=y, y_pred=y_pred, metric=metric)
    return score
    ### ========== TODO : END ========== ###


######################################################################
# main
######################################################################
 
def main() :
    np.random.seed(1234)
    
    # read the tweets and its labels   
    dictionary = extract_dictionary('../data/tweets.txt')
    X = extract_feature_vectors('../data/tweets.txt', dictionary)
    y = read_vector_file('../data/labels.txt')
    
    metric_list = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]
    
    ### ========== TODO : START ========== ###
    # part 1c: split data into training (training + cross-validation) and testing set
    X_train, y_train = X[:560], y[:560]
    X_test, y_test = X[560:], y[560:]

    # part 2b: create stratified folds (5-fold CV)
    kf = StratifiedKFold(y_train, n_folds=5)
    
    # part 2d: for each metric, select optimal hyperparameter for linear-kernel SVM using CV
    # For reference: select_param_linear(X, y, kf, metric="accuracy")

    # for metric in metric_list:
    # 	best_c, c_perf = select_param_linear(X=X_train, y=y_train, kf=kf, metric=metric)
    # 	c_tuples = collections.OrderedDict(sorted(c_perf.items()))
    	
    # 	print '--------------------------------------------------'
    # 	print c_tuples.items()
    # 	print ''
    # 	print 'Best value of C: ' + str(best_c) + ' (' + str(c_tuples[best_c]) + ')'
    # 	print ''

    # part 3c: for each metric, select optimal hyperparameter for RBF-SVM using CV
    # for metric in metric_list:
    # 	best_c_g, best_perf = select_param_rbf(X=X_train, y=y_train, kf=kf, metric=metric)

    # 	print '--------------------------------------------------'
    # 	print ''
    # 	print 'Best value of (C, gamma): ' + '{0}'.format(best_c_g)
    # 	print 'Best score: ' + str(best_perf)
    # 	print ''
    
    # part 4a: train linear- and RBF-kernel SVMs with selected hyperparameters
    # For linear: C = 100.0
    # For RBF: C = 100.0, gamma = 0.01
    linear_C = 100.0
    RBF_C, RBF_G = 100.0, 0.01

    print 'Using C=' + str(linear_C) + ' to train a linear-kernel SVM...'
    linear_svm = SVC(C=linear_C, kernel='linear')
    linear_svm.fit(X_train, y_train)

    print 'Using C=' + str(RBF_C) + ', G=' + str(RBF_G) + ' to train a RBF-kernel SVM'
    rbf_svm = SVC(C=RBF_C, kernel='rbf', gamma=RBF_G)
    rbf_svm.fit(X_train, y_train)

    # part 4c: report performance on test data
    # performance_test(clf, X, y, metric="accuracy")
    for metric in metric_list:
    	linear_score = performance_test(clf=linear_svm, X=X_test, y=y_test, metric=metric)
    	rbf_score = performance_test(clf=rbf_svm, X=X_test, y=y_test, metric=metric)
    	print 'Metric: ' + metric
    	print '------------------'
    	print 'Linear SVM Score: ' + str(linear_score)
    	print 'RBF SVM Score: ' + str(rbf_score)
    	print ''
    
    ### ========== TODO : END ========== ###
    
    
if __name__ == "__main__" :
    main()
