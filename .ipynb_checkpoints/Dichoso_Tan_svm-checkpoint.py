# Name: Aaron Gabrielle C. Dichoso & Tyler Justin Tan
import numpy as np
import random as random

class SupportVectorMachine(object):
    def __init__(self, C, epsilon=1e-4, max_passes=10, kernel_fn="linear", sigma=0.5):
        """
        Inputs:
        - C: (float) Regularization parameter
        - epsilon: (float) numerical tolerance 
        - max_passes: (int) maximum number of times to iterate over alphas without changing
        - kernel: (str) kernel to be used. (linear or rbf)
        - sigma: (float) parameter of the rbf kernel
        """
        self.C = C
        self.epsilon = epsilon
        self.max_passes = max_passes
        self.kernel_fn = kernel_fn
        self.sigma = sigma
        
    def initialize_parameters(self,N):
        """
        Initialize the parameters of the model to zero.
        
        alpha: (float) Lagrangian multipliers; has shape (N,)
        b: (float) scalar; bias term for the hyperplane 

        Input:
        - N: (int) Number of examples in the data set
        """
        
        self.params = {}
        #############################################################################
        # TODO: Initialize the langrangian multipliers and bias. Technically, the   #
        # weight vector does not need to be initialized since it is computed as     #
        # a function of the x's, alpha's and y's. But it is convenient to have the  #
        # parameters to be all in one collection.                                   #
        #############################################################################
        self.params['alpha'] = np.zeros(N)
        self.params['b'] = 0
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################        
        self.params['W'] = 0

    def compute_weights(self,X,y):
        """
        Computes for the weights W. This can be implemented to accomodate both 
        batch and single examples. But for this exercise we do not require you to 
        vectorize your implementations.
        
        Input:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing the ground truth values.
        """
        #############################################################################
        # TODO: Compute for the weights W using the given formula in the notebook.  #
        #############################################################################

        W = np.zeros(X.shape[1])
        for i in range(y.shape[0]):
            W += self.params['alpha'][i] * y[i] * X[i]

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################   
        return W
        

    def f(self,x):
        """
        Computes for the hyperplane. This can be implemented to accomodate both 
        batch and single examples. But for this exercise we do not require you to 
        vectorize your implementations.

        Input:
        - x: A numpy array containing a training example; 
        """

        #############################################################################
        # TODO: Compute for the hyperplane f(x).                                    #
        #############################################################################
        # f = self.params['W'].transpose().dot(x) + self.params['b']
        # f = self.kernel(self.params['W'].transpose(), x) + self.params['b']

        f = 0
        for i in range(self.X.shape[0]):
            f += self.params['alpha'][i] * self.y[i] * self.kernel(self.X[i], x)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################  

        f += self.params['b']
        return f

    def kernel(self,x, z):
        """
        Computes for the correspoding kernel. This can be implemented to accomodate 
        both batch and single examples. But for this exercise we do not require you 
        to vectorize your implementations.

        Input:
        - x: A numpy array containing a training example; 
        """
        #############################################################################
        # TODO: Implement both the linear and gaussian (rbf) kernels.               #
        #############################################################################

        kernel = 0

        if self.kernel_fn == "linear":
            kernel = x.transpose().dot(z) + self.C
        elif self.kernel_fn == "gaussian" or self.kernel_fn =="rbf":
            kernel = np.exp(-np.linalg.norm(x - z)**2/(2*self.sigma**2))

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################  
        return kernel

    def train(self, X, y):
        """
        Train Linear Regression using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing the ground truth values.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        self.X = X
        self.y = y
        N, D = X.shape
        
        self.initialize_parameters(N)
        
        passes = 0
        while passes < self.max_passes:
            print("Current Pass:", str(passes))
            alphas_changed = 0
            
            # iterate through all possible alpha_i's
            for i in range(N):   

                self.params["W"] = self.compute_weights(X, y)

                #############################################################################
                # TODO: Compute for the error E_i between the SVM output and the ith class. #
                #############################################################################
                E_i = self.f(X[i]) - y[i]
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################  

                # Check for KKT conditions
                if (y[i]*E_i < -self.epsilon and self.params['alpha'][i] < self.C) or \
                        (y[i]*E_i > self.epsilon and self.params['alpha'][i] > 0):

                    #############################################################################
                    # TODO: Randomly choose j such that i is not equal to j.                    #
                    #############################################################################
                    while(1):
                        j = random.randrange(0, N)
                        if i != j:
                            break
                    #############################################################################
                    #                              END OF YOUR CODE                             #
                    #############################################################################  
                    
                    #############################################################################
                    # TODO: Compute for the error E_i between the SVM output and the ith class. #
                    #############################################################################
                    E_j = self.f(X[j]) - y[j]
                    #############################################################################
                    #                              END OF YOUR CODE                             #
                    ############################################################################# 
                    
                    alpha_i = self.params['alpha'][i]
                    alpha_j = self.params['alpha'][j]
                    

                    #############################################################################
                    # TODO: Compute for lower and upper bounds. [L, H]                          #
                    #############################################################################

                    if(y[i] != y[j]):
                        L = max(0, alpha_j - alpha_i)
                        H = min(self.C, self.C + alpha_j - alpha_i)
                    else:
                        L = max(0, alpha_i + alpha_j - self.C)
                        H = min(self.C, alpha_i + alpha_j)

                    #############################################################################
                    #                              END OF YOUR CODE                             #
                    ############################################################################# 

                    #############################################################################
                    # TODO: Check if the lower bound and upper bound is the same. Note that     #
                    # these are floating values so we only check if they are the same within    #
                    # some numerical precision. If they are the same then we move on to the     #
                    # next alpha_i.                                                             #
                    #############################################################################
                    
                    if L == H:
                        continue

                    #############################################################################
                    #                              END OF YOUR CODE                             #
                    ############################################################################# 

                    
                    #############################################################################
                    # TODO: Compute for eta using the formula given in the notebook.            #
                    #############################################################################
                    # eta = 2*np.dot(X[i], X[j]) - np.dot(X[i], X[i]) - np.dot(X[j], X[j])
                    eta = 2*self.kernel(X[i], X[j]) - self.kernel(X[i], X[i]) - self.kernel(X[j], X[j])
                    #############################################################################
                    #                              END OF YOUR CODE                             #
                    #############################################################################

                    # prevent division by zero
                    if eta >= 0:
                        continue
                    #############################################################################
                    # TODO: Compute for new value of alpha_j                                    #
                    #############################################################################
                    new_alpha_j = alpha_j - y[j]*(E_i - E_j)/eta
                    #############################################################################
                    #                              END OF YOUR CODE                             #
                    #############################################################################
                    
                    #############################################################################
                    # TODO: Clip the values of alpha_j so that it lies within the acceptable    #
                    # bounds.                                                                   #
                    #############################################################################
                    
                    if new_alpha_j > H:
                        new_alpha_j = H
                    elif new_alpha_j < L:
                        new_alpha_j = L

                    #############################################################################
                    #                              END OF YOUR CODE                             #
                    #############################################################################

                    #############################################################################
                    # TODO: Check if the new alpha_j is the same as its old value within some   #
                    # numerical precision.                                                      #
                    #############################################################################
                    if abs(new_alpha_j - alpha_j) < self.epsilon:
                        continue
                    #############################################################################
                    #                              END OF YOUR CODE                             #
                    #############################################################################

                    # update the parameter alpha_j
                    self.params['alpha'][j] = new_alpha_j

                    
                    #############################################################################
                    # TODO: Compute for new value of alpha_j                                    #
                    #############################################################################
                    new_alpha_i = alpha_i + y[i]*y[j]*(alpha_j - new_alpha_j)
                    #############################################################################
                    #                              END OF YOUR CODE                             #
                    #############################################################################

                    # update the parameter alpha_i
                    self.params['alpha'][i] = new_alpha_i

                    #############################################################################
                    # TODO: Compute for new value of the bias term b.                           #
                    #############################################################################
                    #b1 = self.params['b'] - E_i - y[i]*(new_alpha_i - alpha_i)*np.dot(X[i], X[i]) - y[j]*(new_alpha_j - alpha_j)*np.dot(X[i], X[j])
                    #b2 = self.params['b'] - E_j - y[i]*(new_alpha_i - alpha_i)*np.dot(X[i], X[j]) - y[j]*(new_alpha_j - alpha_j)*np.dot(X[j], X[j])
                    b1 = self.params['b'] - E_i - y[i]*(new_alpha_i - alpha_i)*self.kernel(X[i], X[i]) - y[j]*(new_alpha_j - alpha_j)*self.kernel(X[i], X[j])
                    b2 = self.params['b'] - E_j - y[i]*(new_alpha_i - alpha_i)*self.kernel(X[i], X[j]) - y[j]*(new_alpha_j - alpha_j)*self.kernel(X[j], X[j])
                    if 0 < new_alpha_i < self.C:
                        self.params['b'] = b1
                    elif 0 < new_alpha_j < self.C:
                        self.params['b'] = b2
                    else:
                        self.params['b'] = (b1 + b2)/2
                    #############################################################################
                    #                              END OF YOUR CODE                             #
                    #############################################################################
                    alphas_changed += 1
            if alphas_changed == 0:
                passes += 1
            else:
                passes = 0


        self.params['W'] = self.compute_weights(X, y)

        #############################################################################
        # TODO: Store only the X's, y's, and alpha's that are support vectors       #
        #############################################################################
        support_vectors = []
        for i in range(N):
            if self.params['alpha'][i] > 0:
                support_vectors.append(i)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.X = self.X[support_vectors]
        self.y = self.y[support_vectors]
        self.params['alpha'] = self.params['alpha'][support_vectors]
        
    def predict(self, X):
        """
        Predict labels for test data using this classifier. This can be implemented to 
        accomodate both batch and single examples. But for this exercise we do not 
        require you to vectorize your implementations.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        #############################################################################
        # TODO: Compute for the predictions on the given test data.                 #
        #############################################################################
        prediction = []
        for i in range(X.shape[0]):
            prediction.append(np.sign(self.f(X[i])))
        prediction = np.array(prediction)
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return prediction

