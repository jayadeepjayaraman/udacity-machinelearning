#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import re

enron_data = pickle.load(open("C:/GitHub/ud120-projects/final_project/final_project_dataset.pkl", "r"))

## Finding the number of people in the enron dataset
len(enron_data)


## Finding number of features per person

s = set()
for x in enron_data.itervalues():
    #print x
    s.add(len(x))
print s

## Finding number of POI's (Person of Interest) in the dataset
x= 0
for y in enron_data.itervalues():
    if y['poi'] == True:
        x = x + 1
print (x)
         
         
## How Many POIs Exist?

f = open('C:/GitHub/ud120-projects/final_project/poi_names.txt', 'r')         
poi_names = []
i=0
for line in f:
    if re.match("\(" , line):
        m = re.search("\s(.*)" , line)
        poi_names.append(m.group(0))
    
print len(poi_names)

##What is the total value of the stock belonging to James Prentice?
enron_data['PRENTICE JAMES']['total_stock_value']

## How many email messages do we have from Wesley Colwell to persons of interest?
enron_data['COLWELL WESLEY']['from_this_person_to_poi']

# What’s the value of stock options exercised by Jeffrey Skilling?
enron_data['SKILLING JEFFREY K']['exercised_stock_options']

print enron_data['SKILLING JEFFREY K']['total_payments']
print enron_data['LAY KENNETH L']['total_payments']
print enron_data['FASTOW ANDREW S']['total_payments']


email_array = []
for x in enron_data.values():
    if x['email_address'] <> 'NaN':
        email_array.append(x['email_address'])

print len(email_array)

salary_array = []
for x in enron_data.values():
    if x['salary'] <> 'NaN':
        salary_array.append(x['salary'])

print len(salary_array)        


total_payments_poi_array = []

for x in enron_data.values():
    if x['total_payments'] == 'NaN' and x['poi'] == True:
        total_payments_poi_array.append(x['total_payments'])
        
len(total_payments_poi_array)/146.0
        