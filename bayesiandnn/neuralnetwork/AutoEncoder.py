import numpy as np
import cPickle, gzip
import theano
import theano.tensor as T 
import math
from layers.HiddenLayer import HiddenLayer, LogisticRegression



class AutoEncoder