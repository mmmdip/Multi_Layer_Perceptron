from openpyxl import load_workbook
import numpy as np
import random

# Reading data from excel sheet
data = np.genfromtxt("hw2_q1_test.csv", delimiter=",", skip_header=0)
( nrow, ncol ) = data.shape                       # nrow = number of instances, ncol = number of columns
target = data[ :, ncol - 1 ]                      # target = list of target values

epoch = 100
bias = -1
eta = 0.2
outputs = np.zeros( 4 )

# Weight initialization randomly
w = np.random.randint( 2, size = ncol )
w = np.where( w > 0, 1, -1 )
w = w.astype( float )

print( "Initial weights: ", w )
print()

# Evaluating the perceptron
for e in range( epoch ):
    classified = True
    for i in range( nrow ):
        instance = data[ i, 0 : ncol - 1 ].transpose()
        output = np.dot( instance, w[ 0 : ncol - 1 ] ) + bias * w[ ncol - 1 ]
        if output >= 0:
            output = 1
        else:
            output = -1
        outputs[ i ] = output
        if output != target[ i ]:
            for wi in range( w.size - 1 ):
                w[ wi ] += eta * ( target[ i ] - output ) * instance[ wi ]
            w[ w.size - 1 ] = w[ wi - 1 ] + eta * ( target[ i ] - output ) * bias
            classified = False
        if not e % 2:
            print( "Epoch: ", e )
            print( "Outputs: ", outputs )
            print( "Weights: ", w )
    if classified:
        print( "Classified correctly" )
        break
    print()
    
