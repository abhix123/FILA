# A Sample Carrom Agent to get you started. The logic for parsing a state
# is built in

from thread import *
import time
import socket
import sys
import argparse
import random
import ast
import one_step, Utils

import numpy as np
from scipy import stats 
from bisect import bisect

END_STATE = {"Black_Locations": [], "White_Locations": [], "Red_Location": [], "Score": 0}
INITIAL_STATE = {'White_Locations': [(400, 368), (437, 420), (372, 428), (337, 367), (400, 332),
                                     (463, 367), (463, 433), (400, 468), (337, 433)],
                 'Red_Location': [(400, 400)],
                 'Score': 0,
                 'Black_Locations': [(400, 432), (363, 380), (428, 372),  (370, 350), (430, 350),
                                     (470, 400), (430, 450), (370, 450), (330, 400)]}

QUEEN_PEICE = {'White_Locations': [(400, 368)],
                 'Red_Location': [(400, 400)],
                 'Score': 0,
                 'Black_Locations': []}

ONE_PEICE = {'White_Locations': [(400, 368)],
                 'Red_Location': [],
                 'Score': 0,
                 'Black_Locations': []}

# init action space
a_index_curr = 0
actions = []
nActions = 0
time_step = 0

for x in range(0, 21):
    for angle in range(0, 51):
        for force in range(1, 40):
            actions.append([0.05*x, 0.02*angle,  0.02*force])
nActions = len(actions)
#sarsa parameters
discount = 1

# Parse arguments

parser = argparse.ArgumentParser()

parser.add_argument('-np', '--num-players', dest="num_players", type=int,
                    default=1,
                    help='1 Player or 2 Player')
parser.add_argument('-p', '--port', dest="port", type=int,
                    default=12121,
                    help='port')
parser.add_argument('-rs', '--random-seed', dest="rng", type=int,
                    default=0,
                    help='Random Seed')
parser.add_argument('-c', '--color', dest="color", type=str,
                    default="Black",
                    help='Legal color to pocket')
args = parser.parse_args()


host = '127.0.0.1'
port = args.port
num_players = args.num_players
random.seed(args.rng)  # Important
color = args.color

# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# s.connect((host, port))


# Given a message from the server, parses it and returns state and action


def parse_state_message(state, reward):
    coins = [0.0]*19*2
    white = state["White_Locations"] 
    black = state["Black_Locations"] 
    queen = state["Red_Location"]
    
    for w in range(len(white)):
        coins[2*w] = white[w][0]/800.0
        coins[2*w+1] = white[w][1]/800.0

    for w in range(len(black)):
        coins[18 + 2*w] = black[w][0]/800.0
        coins[18 + 2*w+1] = black[w][1]/800.0

    for w in range(len(queen)):
        coins[36 + 2*w] = queen[w][0]/800.0
        coins[36 + 2*w+1] = queen[w][1]/800.0
    if reward < 0:
        reward = -1
    return coins, reward


# model dimensions
state_size = 19
state_dim = 2
action_dim = 3
nn_input_dim = state_size*state_dim + action_dim # input layer dimensionality
nn_output_dim = 1 # output layer dimensionality
nn_hdim = 30

# Gradient descent parameters (I picked these by hand)
learning_rate = 0.05 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength


# Initialize the parameters to random values. We need to learn these.
np.random.seed(0)
W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
b1 = np.zeros((1, nn_hdim))
W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
b2 = np.zeros((1, nn_output_dim))
# W3 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim2)
# b3 = np.zeros((1, nn_output_dim))
a1 = np.zeros((1, nn_hdim))
# a2 = np.zeros((1, nn_hdim2))

# Forward propagation
def forward(X):
    global a1
    global W1, W2, b1, b2

    z1 = X.dot(W1) + b1  # Hidden layer 1
    a1 = np.tanh(z1)        # Activation
    # print(z1, a1)
    z2 = a1.dot(W2) + b2   # Output Linear layer | 2d matrix
    # a2 = np.tanh(z2) 
    # z3 = a2.dot(W3) + b3
    # print('W2', W2)
    return z2[0][0]  

def backward(error, X):
    global W1, W2, b1, b2
    dW2 = np.dot(a1.T, error)
    db2 = np.sum(error, axis=0, keepdims=True)
    delta2 = error.dot(W2.T) * (1 - np.power(a1, 2))
    dW1 = np.dot(X.T, delta2)
    db1 = np.sum(delta2, axis=0)

    # dW3 = (a2.T).dot(error)
    # db3 = np.sum(error, axis=0, keepdims=True)
    # delta3 = error.dot(W3.T) * (1 - np.power(a2, 2))
    # dW2 = np.dot(a1.T, delta3)
    # db2 = np.sum(delta3, axis=0)
    # delta2 = error.dot(W2.T) * (1 - np.power(a1, 2))
    # dW1 = np.dot(a1.T, delta2)
    # db1 = np.sum(delta3, axis=0)
    
    # Add regularization terms (b1 and b2 don't have regularization terms)
    dW2 += reg_lambda * W2
    dW1 += reg_lambda * W1
    # Gradient descent parameter update
    W1 += -learning_rate * dW1
    b1 += -learning_rate * db1
    W2 += -learning_rate * dW2
    b2 += -learning_rate * db2

    
# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent

def gameEnd(state):
    return len(state["White_Locations"]) + len(state["Black_Locations"]) + len(state["Red_Location"]) == 0

Qcurr = 0.0
    
def agent_1player():
    maxGames = 50
    game = 1
    nActions = len(actions)

    while game <= maxGames:
        time_step = 0
        coins = ONE_PEICE
        state, reward = parse_state_message(coins, 0)

        maxTime = 500
        # print("Starting game.............", game)
        while time_step < maxTime: 

            # get Q value for every action
            Qs = []
            scores = np.zeros(nActions)
            for a in range(nActions):
                X = np.array(state + actions[a])
                Q = forward(X)
                Qs.append(Q)
                scores[a] = Q
            
            # exp_scores = np.exp(scores)
            # # get probability distribution
            # print "exp scores sum", np.sum(exp_scores)
            probs = scores / np.sum(scores)
            cdf = [probs[0]]
            for i in range(1, len(probs)):
                cdf.append(cdf[-1] + probs[i])

            a_index_curr = bisect(cdf,random.random())
            # print 'prob sum', np.sum(probs)
            # for p in probs:
            #     print p
            # pick action
            # action_indices = np.arange(nActions)
            # a_index_curr = stats.rv_discrete(values=(action_indices, probs)).rvs()
            Qcurr = Qs[a_index_curr]
            action_picked = actions[a_index_curr]
            angle = -45 + action_picked[1]*270
            action = [action_picked[0], angle, action_picked[2]]
            # print 'action', action

            # simulate
            coins, reward = one_step.simulate(coins, one_step.validate(action, coins))
            if gameEnd(coins):
                break
            nextState, reward = parse_state_message(coins, reward)
            # print(nextState, reward)

            # greedy for action of next state
            Qmax,  a_index_next = -np.inf, -1
            for a in range(nActions):
                X = np.array(nextState + actions[a])
                Qnext = forward(X)
                if Qnext > Qmax:
                    a_index_next = a
                    Qmax = Qnext
            
            # update by sarsa equation
            update = reward + discount*Qmax - Qcurr
             
            
            # backpropogate
            X = np.array(state + actions[a_index_curr])
            X = X.reshape(1, nn_input_dim)
            backward(np.array([[update]]), X)
            
            # next step 
            time_step = time_step + 1
            state = nextState

        print time_step
        game = game + 1

    model = {'W1' : W1, 'W2' : W2, 'b1': b1, 'b2': b2}
    np.save('model.npy', model)
    # Load
    read_dictionary = np.load('model.npy').item()


def agent_2player(state, color):

    flag = 1

   
    a = str(random.random()) + ',' + \
        str(random.randrange(-45, 225)) + ',' + str(random.random())

    try:
        s.send(a)
    except Exception as e:
        print "Error in sending:",  a, " : ", e
        print "Closing connection"
        flag = 0

    return flag


agent_1player()
