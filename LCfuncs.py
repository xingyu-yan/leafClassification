import numpy as np
import matplotlib.pyplot as plt

def cgbt(theta,X,y,input_layer_size,hidden_layer_size,num_labels,lamb,alpha,beta,iterN,tol):
    # cgbt: Conjugate gradient descent method with backtracking line search
    # Input:
        # theta: Initial value
        # X: Training data (input)
        # y: Training data (output)
        # input_layer_size / hidden_layer_size / num_labels: As defined in neural network
        # lamb: Regularization variable
        # alpha: Parameter for line search, denoting the cost function will be descreased by 100xalpha percent
        # beta: Parameter for line search, denoting the "step length" t will be multiplied by beta
        # iterN: Maximum number of iterations
        # tol: The procedure will break if the square of the Newton decrement is less than the threshold tol

    # Initialize the gradient
    dxPrev = -nnGrad(theta,input_layer_size,hidden_layer_size,num_labels,X,y,lamb)
    snPrev = dxPrev
    theta = np.matrix(theta).T
    # Iteration
    for i in range(iterN):
        J,grad = nnCostFunction(theta,input_layer_size,hidden_layer_size,num_labels,X,y,lamb)
        dx = -grad
        if dx.T*dx < tol:
            print ('Terminated due to stopping condition with iteration number',i)
            return theta
        # betaPR since beta is already used as backtracking variable
        # Polak-Ribiere formula
        betaPR = np.max((0,(dx.T*(dx-dxPrev))/(dxPrev.T*dxPrev)))
        # Search direction
        sn = np.array(dx+snPrev*betaPR)
        # Backtracking
        t = 1
        costNew = nnCost(theta+t*sn,input_layer_size,hidden_layer_size,num_labels,X,y,lamb)
        alphaGradSn = alpha*(grad.T*sn)
        while costNew > J+t*alphaGradSn or np.isnan(costNew):
            t = beta*t
            costNew = nnCost(theta+t*sn,input_layer_size,hidden_layer_size,num_labels,X,y,lamb)

        tRight = t*2
        tTemp = t
        while tRight-t > 1e-3: # Search right-hand side
            costRight = nnCost(theta+tRight*sn,input_layer_size,hidden_layer_size,num_labels,X,y,lamb)
            if costRight > costNew:
                tRight = (t+tRight)/2
            else:
                t = tRight
                tRight = 2*t
                costNew = costRight

        if t == tTemp:
            tLeft = t/2.0
            while t-tLeft > 1e-3: # Search left-hand side
                costLeft = nnCost(theta+tLeft*sn,input_layer_size,hidden_layer_size,num_labels,X,y,lamb)
                if costLeft > costNew:
                    tLeft = (t+tLeft)/2
                else:
                    t = tLeft
                    tLeft = t/2
                    costNew = costLeft

        # Update
        theta += t*sn
        snPrev = sn
        dxPrev = dx
        print ('Iteration',i+1,' | Cost:',costNew)

    return theta

    
def displayData(X,nameFig):
    # python translation of displayData.m from coursera
    # For now, only "quadratic" image
    example_width = np.round(np.sqrt(X.shape[1]))
    example_height = example_width
    
    display_rows = np.floor(np.sqrt(X.shape[0]))
    display_cols = np.ceil(X.shape[0]/display_rows)

    pad = 1

    display_array = -np.ones((pad+display_rows*(example_height+pad), pad+display_cols*(example_width+pad)))

    curr_ex = 0

    for j in range(display_rows.astype(np.int16)):
        for i in range(display_cols.astype(np.int16)):
            if curr_ex == X.shape[0]:
                break
            max_val = np.max(np.abs(X[curr_ex,:]))
            rowStart = pad+j*(example_height+pad)
            colStart = pad+i*(example_width+pad)
            display_array[rowStart:rowStart+example_height, colStart:colStart+example_width] = X[curr_ex,:].reshape((example_height,example_width)).T/max_val

            curr_ex += 1
        if curr_ex == X.shape[0]:
            break

    plt.imshow(display_array,extent = [0,10,0,10])
    plt.savefig(nameFig)
    plt.show()

def sigmoidGradient(z):
    g = 1/(1+np.exp(-z))
    return np.multiply(g,1-g)

def nnCost(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lamb):
    Theta1 = np.matrix(np.reshape(nn_params[:hidden_layer_size*(input_layer_size+1)],(hidden_layer_size,input_layer_size+1),order='F'))
    Theta2 = np.matrix(np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):],(num_labels,hidden_layer_size+1),order='F'))
    
    a1 = np.c_[np.ones((X.shape[0],1)),X]
    a2 = np.c_[np.ones((X.shape[0],1)),sigmoid(a1*Theta1.T)]
    a3 = sigmoid(a2*Theta2.T)

    Y = np.zeros((X.shape[0],num_labels))

    for i in range(num_labels):
        for j in range(X.shape[0]):
            if y[j] == i+1: # To be consistant with matlab program
                Y[j,i%10] = 1

    J = (np.multiply(-Y,np.log(a3))-np.multiply(1-Y,np.log(1-a3))).sum().sum()/X.shape[0]
    J += lamb*(np.power(Theta1[:,1:],2).sum().sum()+np.power(Theta2[:,1:],2).sum().sum())/X.shape[0]/2
    return J
    
def nnGrad(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lamb):
    Theta1 = np.matrix(np.reshape(nn_params[:hidden_layer_size*(input_layer_size+1)],(hidden_layer_size,input_layer_size+1),order='F'))
    Theta2 = np.matrix(np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):],(num_labels,hidden_layer_size+1),order='F'))
    
    Delta1 = np.zeros((hidden_layer_size,input_layer_size+1))
    Delta2 = np.zeros((num_labels,hidden_layer_size+1))

    for t in range(X.shape[0]):
        # 1
        a_1 = np.matrix(np.r_[np.ones(1),X[t,:].T]).T
        z_2 = np.dot(Theta1,a_1)
        a_2 = np.matrix(np.r_[np.ones((1,1)),sigmoid(z_2)])
        z_3 = np.dot(Theta2,a_2)
        a_3 = sigmoid(z_3)

        # 2
        yvec = np.zeros((num_labels,1))
        yvec[y[t]-1] = 1
        delta3 = a_3-yvec

        # 3
        delta2 = np.multiply(Theta2.T*delta3,np.matrix(np.r_[np.ones((1,1)),sigmoidGradient(z_2)]))

        # 4
        delta2 = delta2[1:]
        Delta2 += delta3*a_2.T
        Delta1 += delta2*a_1.T

    Theta1_grad = Delta1/X.shape[0]
    Theta2_grad = Delta2/X.shape[0]

    Theta1_grad[:,1:] = Theta1_grad[:,1:]+Theta1[:,1:]*lamb/X.shape[0]
    Theta2_grad[:,1:] = Theta2_grad[:,1:]+Theta2[:,1:]*lamb/X.shape[0]
    
    grad = np.r_[np.matrix(np.reshape(Theta1_grad,Theta1.shape[0]*Theta1.shape[1],order='F')).T,np.matrix(np.reshape(Theta2_grad,Theta2.shape[0]*Theta2.shape[1],order='F')).T]

    return grad

def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lamb):
    Theta1 = np.matrix(np.reshape(nn_params[:hidden_layer_size*(input_layer_size+1)],(hidden_layer_size,input_layer_size+1),order='F'))
    Theta2 = np.matrix(np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):],(num_labels,hidden_layer_size+1),order='F'))
    
    a1 = np.c_[np.ones((X.shape[0],1)),X]
    a2 = np.c_[np.ones((X.shape[0],1)),sigmoid(a1*Theta1.T)]
    a3 = sigmoid(a2*Theta2.T)

    Y = np.zeros((X.shape[0],num_labels))

    for i in range(num_labels):
        for j in range(X.shape[0]):
            if y[j] == i+1: # To be consistant with matlab program
                Y[j,i%10] = 1

    J = (np.multiply(-Y,np.log(a3))-np.multiply(1-Y,np.log(1-a3))).sum().sum()/X.shape[0]
    J += lamb*(np.power(Theta1[:,1:],2).sum().sum()+np.power(Theta2[:,1:],2).sum().sum())/X.shape[0]/2

    Delta1 = np.zeros((hidden_layer_size,input_layer_size+1))
    Delta2 = np.zeros((num_labels,hidden_layer_size+1))

    for t in range(X.shape[0]):
        # 1
        a_1 = np.matrix(np.r_[np.ones(1),X[t,:].T]).T
        z_2 = np.dot(Theta1,a_1)
        a_2 = np.matrix(np.r_[np.ones((1,1)),sigmoid(z_2)])
        z_3 = np.dot(Theta2,a_2)
        a_3 = sigmoid(z_3)

        # 2
        yvec = np.zeros((num_labels,1))
        yvec[y[t]-1] = 1
        delta3 = a_3-yvec

        # 3
        delta2 = np.multiply(Theta2.T*delta3,np.matrix(np.r_[np.ones((1,1)),sigmoidGradient(z_2)]))

        # 4
        delta2 = delta2[1:]
        Delta2 += delta3*a_2.T
        Delta1 += delta2*a_1.T

    Theta1_grad = Delta1/X.shape[0]
    Theta2_grad = Delta2/X.shape[0]

    Theta1_grad[:,1:] = Theta1_grad[:,1:]+Theta1[:,1:]*lamb/X.shape[0]
    Theta2_grad[:,1:] = Theta2_grad[:,1:]+Theta2[:,1:]*lamb/X.shape[0]
    
    grad = np.r_[np.matrix(np.reshape(Theta1_grad,Theta1.shape[0]*Theta1.shape[1],order='F')).T,np.matrix(np.reshape(Theta2_grad,Theta2.shape[0]*Theta2.shape[1],order='F')).T]

    return J,grad

def sigmoid(z):
    return 1/(1+np.exp(-z))

def predict(Theta1,Theta2,X):
    X = np.c_[np.ones((X.shape[0],1)),X]
    z2 = np.dot(Theta1,X.T)
    a2 = np.r_[np.ones((1,z2.shape[1])),sigmoid(z2)]
    z3 = np.dot(Theta2,a2)
    a3 = sigmoid(z3)
    p = np.zeros((X.shape[0],1))
    for i in range(a3.shape[1]):
        for j in range(a3.shape[0]):
            if a3[j,i] == np.max(a3[:,i]):
                p[i] = j+1
    return p
            
def randInitializeWeights(L_in,L_out):
    epsilon_init = 0.12
    return np.random.rand(L_out,L_in+1)*2*epsilon_init-epsilon_init

def debugInitializeWeights(fan_out,fan_in):
    num_el = fan_out*(fan_in+1)
    return np.reshape(np.sin(np.linspace(1,num_el,num_el)),(fan_out,fan_in+1),order='F')/10

def computeNumericalGradient(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lamb):
    numgrad = np.zeros(nn_params.shape)
    perturb = np.zeros(nn_params.shape)
    e = 1e-4
    for p in range(nn_params.shape[0]):
        perturb[p] = e
        loss1 = nnCostFunction(nn_params-perturb,input_layer_size,hidden_layer_size,num_labels,X,y,lamb)
        loss2 = nnCostFunction(nn_params+perturb,input_layer_size,hidden_layer_size,num_labels,X,y,lamb)
        numgrad[p] = (loss2[0]-loss1[0])/2/e
        perturb[p] = 0
    return numgrad

def checkNNGradients(lamb):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    Theta1 = debugInitializeWeights(hidden_layer_size,input_layer_size)
    Theta2 = debugInitializeWeights(num_labels,hidden_layer_size)
    X = debugInitializeWeights(m,input_layer_size-1)
    y = np.mod(np.linspace(1,m,m),num_labels)+1
    
    nn_params = np.r_[np.matrix(np.reshape(Theta1, Theta1.shape[0]*Theta1.shape[1], order='F')).T,np.matrix(np.reshape(Theta2, Theta2.shape[0]*Theta2.shape[1], order='F')).T]
    (cost,grad) = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lamb)
    numgrad = computeNumericalGradient(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lamb)
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
#    print(cost)
    print ('If your backpropagation implementation is correct, then the relative difference will be small (less than 1e-9). \nRelative difference:',diff,'\n')
    
def print_results(name, errors):
    errors_square = np.zeros(errors.shape)
    for t in range(errors.shape[0]):
        for tt in range(errors.shape[1]):
            abcde = errors[t,tt]**2   
            errors_square[t,tt] = abcde 
    mae = np.mean(np.mean(np.abs(errors)))
    rmse = np.sqrt(np.mean(np.mean(errors_square)))
    print(name,' nMAE is %f\n', mae)
    print(name,' nRMSE is %f\n', rmse)