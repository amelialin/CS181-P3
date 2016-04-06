import numpy as np
import csv
import os
import pandas as pd

# Predict via the user-specific median.
# If the user has no data, use the global median.

train_file = 'train.csv'
test_file  = 'test_small.csv'
soln_file  = 'matrix_factorization_results.csv'

###################################################################
#set local drive
# os.chdir("C:\Users\WouterD\Dropbox\Recent Work Wouter\Practical3")
####################################################################

# Load the training data.
num_rows = 100 # only loads subset of data for speed

print "Loading training data..."
train_data = {}
with open(train_file, 'r') as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    next(train_csv, None)
    for row in train_csv:
        user   = row[0]
        artist = row[1]
        plays  = row[2]
    
        if not user in train_data:
            train_data[user] = {}
        
        train_data[user][artist] = int(plays)

        num_rows -= 1
        if num_rows == 0:
            break

###################################################################
####Starting Small - 10 pairs
first10pairs = {k: train_data[k] for k in train_data.keys()[:10]}
# print "first10pairs", first10pairs
testpd=pd.DataFrame(first10pairs)
# print "testpd", testpd
testpd=testpd.fillna(0)
column_names = list(testpd.columns.values) # get column names (users)
row_names = list(testpd.index) # get row names (artists)
# print "column_names", column_names
# print "row_names", row_names
R=np.array(testpd)
# print "R", R


###############################################################################
##Matrix Factorization
"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""
def matrix_factorization(R, P, Q, K, steps=1000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in xrange(steps):
        # print "Step:", step
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P,Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    return P, Q.T

###############################################################################

#MINI TEST
############################################################################

N = len(R)
M = len(R[0])
K = 2

P = np.random.rand(N,K)
Q = np.random.rand(M,K)

print "Doing matrix factorization..."
nP, nQ = matrix_factorization(R, P, Q, K)
# print "nP", nP
# print "nP.shape", nP.shape
# print "nQ", nQ
# print "nQ.shape", nQ.shape
nR=np.dot(nP,nQ.T)
# print "nR", nR
# print "nR.shape", nR.shape

###############################################################################
#OLD CODE - COMMENTED OUT

# Compute the global median and per-user median.
#plays_array  = []
#user_medians = {}
#for user, user_data in train_data.iteritems():
#    user_plays = []
#    for artist, plays in user_data.iteritems():
#        plays_array.append(plays)
#        user_plays.append(plays)
#
#    user_medians[user] = np.median(np.array(user_plays))
#global_median = np.median(np.array(plays_array))

# Compute the per-user median.
plays_array  = []
user_medians = {}
for user, user_data in train_data.iteritems():
   user_plays = []
   for artist, plays in user_data.iteritems():
       plays_array.append(plays)
       user_plays.append(plays)

   user_medians[user] = np.median(np.array(user_plays))
global_median = np.median(np.array(plays_array))


#########
###Re-integrate the new matrix with the old dataframe columns and headings. Column labels are users, row labels are artists.
nR_pd=pd.DataFrame(nR, columns=column_names, index=row_names)
print "nR_pd", nR_pd
print "nR_pd.shape", nR_pd.shape
print "column_names", column_names
print "row_names", row_names


##############################################################################
###THIS DOESNT WORK YET


# Write out test solutions.
with open(test_file, 'r') as test_fh:
    test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
    next(test_csv, None)

    with open(soln_file, 'w') as soln_fh:
        soln_csv = csv.writer(soln_fh,
                              delimiter=',',
                              quotechar='"',
                              quoting=csv.QUOTE_MINIMAL)
        soln_csv.writerow(['Id', 'plays'])

        for row in test_csv:
            id     = row[0]
            user   = row[1]
            artist = row[2]

            if (str(user) in column_names) & (str(artist) in row_names):
                # print "We got it!"
                # plays = nR_pd.loc[str(artist), str(user)]
                # print plays
                soln_csv.writerow([id, nR_pd.loc[str(artist), str(user)]])
            else:
                print "User or artist with id", id, "not in training data."
                # soln_csv.writerow([id, user_medians[user]])
                soln_csv.writerow([id, global_median])
            # plays = nR_pd.loc[str(artist), str(user)]
            # print plays

            # if user in user_medians:
            #     soln_csv.writerow([id, user_medians[user]])
            # else:
            #     print "User", id, "not in training data."
            #     soln_csv.writerow([id, global_median])
                
