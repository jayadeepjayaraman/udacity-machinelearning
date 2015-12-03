#!/usr/bin/python
import numpy


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    
    residual = 0.0
    data = ()    
    for ix in numpy.ndindex(*predictions.shape):   
        residual =   abs(predictions [ix]- net_worths[ix])
        data = (ages[ix], net_worths[ix], float(residual))
        cleaned_data.append(data)        
    
    cleaned_data = sorted(cleaned_data, key = lambda x: x[2])
    percent_limit = int(len(cleaned_data) *0.9)
    
    cleaned_data = cleaned_data[0:percent_limit]
            
    return cleaned_data

