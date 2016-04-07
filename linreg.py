import numpy as np
import csv

# Predict via the median number of plays.

train_file = 'train.csv'
user_file = 'profiles.csv'
count_file = 'countries.csv'
test_file  = 'test.csv'
soln_file  = 'solution.csv'

#set debug to false to iterate over the entire train_data and test_data files
debug = True
train_inputs = 1000
test_inputs = 1000

# Load the training data.
train_data = {}
iterate = 0
with open(train_file, 'r') as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    next(train_csv, None)
    for row in train_csv:
        user   = row[0]
        artist = row[1]
        plays  = int(row[2])
        
        #Create a new dict if user isn't in list
        if not user in train_data:
            train_data[user] = {}
            
            #convert user ID to a number            
            uID = 0
            for char in user:
                uID = uID + ord(char)
            train_data[user]["uID"] = uID
        
        #convert arist ID to a number  
        aID = 0
        for char in artist:
            aID = aID + ord(char)
        train_data[user][artist] = {}
        train_data[user][artist]["plays"] = plays
        train_data[user][artist]["aID"] = aID
        
        #DEBUG: code to select train_data size
        if debug == True:        
            iterate += 1
            if iterate == train_inputs:
                break

#load the countries array        
countries = {}
with open(count_file, 'r') as count_fh:
    count_csv = csv.reader(count_fh, delimiter=',', quotechar='"')
    next(count_csv, None)
    for row in count_csv:
        country = row[0]
        c_id = int(row[1])
        
        countries[country] = c_id

#load the user info
users = {}
with open(user_file, 'r') as train_prof:
    profile_csv = csv.reader(train_prof, delimiter=',', quotechar='"')
    next(profile_csv, None)
    for row in profile_csv:
        user = row[0]
        
        #assign values to gender based on number of x-chromosomes
        if row[1] == 'm':
            sex = 1
        elif row[1] == 'f':
            sex = 2
        else:
            sex = 0
        
        #set age to zero if blank
        if row[2] != '':        
            age = int(row[2])
        else:
            age = 0
        
        #convert country to int.
        country = countries[row[3]]
        
        #place the extra data in train_data
        if user in train_data:
            train_data[user]['Sex'] = sex
            train_data[user]['Age'] = age
            train_data[user]['Country'] = country
        
        #additinally fill a user array
        users[user] = {}
        users[user]['Sex'] = sex
        users[user]['Age'] = age
        users[user]['Country'] = country

#format the inputs for our regression
X = []
Y = []
ind = 0
for user, user_data in train_data.iteritems():
    for artist, artist_data in user_data.iteritems():
        if type(artist_data) is dict:
            X.append([])
            X[ind].append(user_data['Age'])
            X[ind].append(user_data['Sex'])
            X[ind].append(user_data['Country'])
            X[ind].append(user_data['uID'])
            X[ind].append(artist_data['aID'])
            Y.append(artist_data['plays'])
            ind += 1
        
#fit
w = np.linalg.lstsq(np.array(X), np.array(Y))        

#read the test_file in as just numbers
inData = []
iterate = 0
with open(test_file, 'r') as test_fh:
    test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
    next(test_csv, None)
    for row in test_csv:
        user = row[1]
        artist = row[2]
    
        inData.append([])
        inData[iterate].append(users[user]['Age']) 
        inData[iterate].append(users[user]['Sex']) 
        inData[iterate].append(users[user]['Country']) 
        
        uID = 0
        for char in user:
            uID = uID + ord(char)
        inData[iterate].append(uID) 
    
        aID = 0
        for char in artist:
            aID = aID + ord(char)        
        inData[iterate].append(aID)
        
        #DEBUG code to select number of inputs
        if debug == True:
            iterate += 1
            if iterate == test_inputs:
                break

#predict
print np.array(inData).shape
print w[0].shape
print inData[0]
results = np.dot(np.array(inData), w[0])

# Write out the solutions.
with open(soln_file, 'wb') as soln_fh:
    soln_csv = csv.writer(soln_fh,
                          delimiter=',',
                          quotechar='"',
                          quoting=csv.QUOTE_MINIMAL)
    soln_csv.writerow(['Id', 'plays'])
    
    index = 1   
    for result in results:
        soln_csv.writerow([index, result])
        index += 1