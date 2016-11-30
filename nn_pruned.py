# A Sample Carrom Agent to get you started. The logic for parsing a state
# is built in

from thread import *
import math
import time
import socket
import sys
import argparse
import random
import ast
from math import sqrt, sin, cos, tan
import one_step, Utils

import numpy as np
from scipy import stats 
from bisect import bisect

# Parse arguments


STRIKER_RADIUS = 20.6
POCKET_RADIUS = 22.51
POCKETS = [(44.1, 44.1), (755.9, 44.1), (755.9, 755.9), (44.1, 755.9)]
STRIKER_Y = 140
STRIKER_XL, STRIKER_XR = 170, 630
CLUSTER_RADIUS_1P = 6*STRIKER_RADIUS
CLUSTER_RADIUS = 4*STRIKER_RADIUS
CLUSTER_MAX_DIST = 15*POCKET_RADIUS
CLUSTER_MIN_SIZE = 2
CLUSTER_MIN_SIZE2 = 2
COIN_RADIUS = 15.01

BOARD_SIZE = 800
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

TWO_PEICE = {'White_Locations': [(400, 368), (200, 560)],
                 'Red_Location': [],
                 'Score': 0,
                 'Black_Locations': []}

# init action space
a_index_curr = 0
actions = []
nActions = 0
time_step = 0



def angle(a, b):
    ax, ay = a
    bx, by = b

    if bx == ax:
        if by > ay:
            return 90
        else: 
            return -90

    theta = math.degrees(math.atan((by-ay)/(bx-ax)))
    if bx<ax :
        return theta
    else: 
        return 180 + theta 

def dist(A,B):
    ax, ay = A
    bx, by = B
    return math.hypot(bx-ax, by-ay)

def intersection(coin, pocket):
    ax, ay = coin
    bx, by = pocket
    if not same_side(ay, by):      # check if coin lies vetween striker and pocket
        return -1
    if by == ay:
        return -1
    x = (ax - (ay - STRIKER_Y)*((bx - ax)/(by - ay)))
    return x

def intersection2(coin, angle):
    x = coin[0] - (coin[1] - STRIKER_Y)/math.tan(math.radians(angle))
    return min(STRIKER_XR, max(STRIKER_XL, x))

def same_side(y1, y2):
    return ((y1 > STRIKER_Y) and (y2 > STRIKER_Y)) or ((y1 < STRIKER_Y) and (y2 < STRIKER_Y))

def angle_vectors(a,b):
    ax,ay=a
    bx,by=b
    angle = math.degrees(math.acos((ax*bx+ay*by)/(sqrt(ax*ax+ay*ay)*sqrt(bx*bx+by*by))))
    return angle


# Check for obstructions
 
def isBlocked(striker, coins):
    for c in coins:
        if (dist(striker, c)  <= (STRIKER_RADIUS + COIN_RADIUS)):
            return True
    return False 

def obstructs(striker,pocket,coin):
    sx, sy = striker
    px, py = pocket
    cx, cy = coin
    x = (cx*pow(px-sx,2) + sx*pow(py-sy,2) - (sy-cy)*(py-sy)*(px-sx))/(pow(px-sx,2)+pow(py-sy,2))
    y = ((py-sy)/(px-sx))*(x-sx) + sy
    if((x<sx and x>px) or (x<px and x>sx)):
        if dist(coin,[x,y]) < COIN_RADIUS + STRIKER_RADIUS:
            return 1
    return 0;

def obstructs_in_list(A, B, given_list):
    for elem in given_list:
        if elem != A:
            if obstructs(A,B,elem):
                return True
    return False

####

def getShootingLine(coin, pocket):
    cx,cy = coin
    px,py = pocket
    if (not same_side(cy, py)):  # striker cannot lie between coin and pocket
        return -1,-1

    delta = 4*POCKET_RADIUS
    x = intersection(coin, pocket)
    x = min(STRIKER_XR - delta , max(STRIKER_XL + delta, x))
    slope = angle(coin, (x, STRIKER_Y))
    if slope >= -45.0 and slope <= 225.0:
        return x, slope
    
    return -1,-1


def cluster(coin, our_coins, their_coins):
    sz1, sz2, cx, cy = 0, 0, 0, 0
    for s in our_coins:
        if dist(coin, s) < CLUSTER_RADIUS:
            sz1 = sz1 + 1
            cx += s[0]
            cy += s[1]
    cy /= sz1
    cx /= sz1
    for s in their_coins:
        if(dist(coin,s)<CLUSTER_RADIUS):
            sz2 = sz2 + 1

    if dist((cx, cy), coin) < 2*COIN_RADIUS:
        return sz1, sz2, cx, cy
    return sz1, sz2, coin[0], coin[1]

def cluster_1p(coin, state):
    sz, cx, cy = 0, 0, 0
    for s in state:
        if dist(coin, s) < CLUSTER_RADIUS_1P:
            sz = sz + 1
            cx += s[0]
            cy += s[1]
    cy /= sz
    cx /= sz
    if dist((cx, cy), coin) < 2*COIN_RADIUS:
        return sz, cx, cy
    return sz, coin[0], coin[1]



def closest_pocket(coin):
    pd, pocket = float('inf'), -1
    for p in range(len(POCKETS)):
        if same_side(coin[1], POCKETS[p][1]) and (dist(coin, POCKETS[p]) < pd):
            pocket = p
            pd = dist(coin, POCKETS[p])
    return pocket, pd

def get_priority(sz1, sz2, pd, code):
    if code == 0:
        if sz1 < CLUSTER_MIN_SIZE or pd > CLUSTER_MAX_DIST:
            return -1
        return (sz1 - sz2)/pd
    elif code == 1:
        if sz1 < CLUSTER_MIN_SIZE2:
            return -1
        return sz1 - sz2

def get_priority_1p(sz, pd, code):
    if code == 0:
        if sz < CLUSTER_MIN_SIZE or pd > CLUSTER_MAX_DIST:
            return -1
        return sz/pd
    elif code == 1:
        if sz < CLUSTER_MIN_SIZE2:
            return -1
        return sz


# code : 0   pocket distance matters
# code : 1   only cluster size

def select_cluster(our_coins, their_coins, code):
    max_priority = 0
    cx, cy, poc = 0, 0, -1
    striker_x = -1
    priority_sz = -1
    delta = 2*POCKET_RADIUS
    for s in our_coins:
        sz1, sz2, x, y = cluster(s, our_coins, their_coins)
        p, pd = closest_pocket(s)
        priority = get_priority(sz1, sz2, pd, code)
        if priority > max_priority :
            #striker_x, slope =  getShootingLine((x, y), POCKETS[p])
            # if striker_x - delta > STRIKER_XL:
            #     striker_x -= delta
            # else:
            #     striker_x += delta
            striker_x, striker_angle = getX_cut_shot((x, y), POCKETS[p], our_coins + their_coins)
            if striker_x < 0:
                continue
            cx, cy, poc = x, y, p
            max_priority = priority
            priority_sz = sz1


    
    # print(" cluster centre ", cx, cy, priority_sz)  
    sd = dist((cx, cy), (striker_x, STRIKER_Y)) 
    pd = dist((cx, cy), POCKETS[poc]) 
    force = 0
    if priority_sz < 5:
        force = 0.05 + 0.05*priority_sz*((2*pd + sd)/BOARD_SIZE) 
    else:
        #force = 1.0 #0.05*sz*((sd + math.sqrt(pd))/BOARD_SIZE)
        force = min((0.2 + pd * pd * priority_sz/(BOARD_SIZE * BOARD_SIZE)), 1.0)  

    #if len(coins) >= 10 and cy > STRIKER_Y + 50:
       # force = 1.0
   
    if striker_x >= 0:
        return striker_x, striker_angle, priority_sz, force   
    
    return -1, -1, -1, -1

def select_cluster_1p(coins, code):
    max_priority = 0
    cx, cy, poc = 0, 0, -1
    striker_x = -1
    priority_sz = -1
    delta = 2*POCKET_RADIUS
    for s in coins:
        sz, x, y = cluster_1p(s, coins)
        p, pd = closest_pocket(s)
        priority = get_priority_1p(sz, pd, code)
        if priority > max_priority :
            #striker_x, slope =  getShootingLine((x, y), POCKETS[p])
            # if striker_x - delta > STRIKER_XL:
            #     striker_x -= delta
            # else:
            #     striker_x += delta
            striker_x, striker_angle = getX_cut_shot((x, y), POCKETS[p], coins)
            if striker_x < 0:
                continue
            cx, cy, poc = x, y, p
            max_priority = priority
            priority_sz = sz


    
    # print(" cluster centre ", cx, cy, priority_sz)  
    sd = dist((cx, cy), (striker_x, STRIKER_Y)) 
    pd = dist((cx, cy), POCKETS[poc]) 
    force = 0
    if sz < 5:
        force = 0.05 + 0.05*priority_sz*((2*sd + pd)/BOARD_SIZE) 
    else:
        force = 1.0 #0.05*sz*((sd + math.sqrt(pd))/BOARD_SIZE) 

    if len(coins) >= 10 and cy > STRIKER_Y + 50:
        force = 1.0

    if striker_x >= 0:
        return striker_x, striker_angle, priority_sz, force   
    
    return -1, -1, -1, -1

def getModifiedCoinCoords(coin , pocket):
    modified_coin_x= coin[0] + (coin[0]-pocket[0])*(STRIKER_RADIUS+COIN_RADIUS)/dist(coin,pocket);
    modified_coin_y= coin[1] + (coin[1]-pocket[1])*(STRIKER_RADIUS+COIN_RADIUS)/dist(coin,pocket);
    return [modified_coin_x, modified_coin_y]


def getX_cut_shot( target_coin, pocket, coins):
    striker_pos, striker_angle = -1, -1
    min_score = float("inf")

    for striker_x in range(STRIKER_XL, STRIKER_XR, 10):                
        angle_wrt_base= angle(target_coin,[striker_x,STRIKER_Y])

        if angle_wrt_base < -45 or angle_wrt_base > 225: 
            continue

        if isBlocked([striker_x,STRIKER_Y], coins):
            continue

        projection_angle= angle_vectors([target_coin[0]-striker_x,target_coin[1]-STRIKER_Y],[pocket[0]-target_coin[0],pocket[1]-target_coin[1]])

        if(projection_angle < 0):
            continue

        if( projection_angle < min_score):
            min_score = projection_angle
            striker_pos = striker_x
            striker_angle = angle_wrt_base


    return striker_pos, striker_angle

def cut_shot(our_coins, their_coins, code, queen = [-1,-1]):
    cut_shot_possible=False
    
    striker_pos = -1
    striker_angle =-1
    min_score = float("inf")
    target_coin = -1, -1
    target_pocket = -1, -1
    force = -1
    modified_coins = []
    factor = 1.0
    if queen == [-1,-1]:
        modified_coins = our_coins
    else:
        modified_coins = [queen]

    for coin in modified_coins:
        for pocket in POCKETS:
            coin_to_pocket_obstruction = obstructs_in_list(coin, pocket, our_coins + their_coins)
            if coin_to_pocket_obstruction:
                continue
            
            for i in range(0,50):
                    
                xpos=0.02*i

                striker_x= 170+xpos*460
                
                #if len(coins) < 5:
                    #factor = 0.8
                modified_coin_x= coin[0] + factor*(coin[0]-pocket[0])*(STRIKER_RADIUS+COIN_RADIUS)/dist(coin,pocket);
                modified_coin_y= coin[1] + factor*(coin[1]-pocket[1])*(STRIKER_RADIUS+COIN_RADIUS)/dist(coin,pocket);
                modified_coin = [modified_coin_x, modified_coin_y]



                projection_angle= angle_vectors([modified_coin_x-striker_x,modified_coin_y-STRIKER_Y],[pocket[0]-coin[0],pocket[1]-coin[1]])

                if code:
                    if (projection_angle) > 60 or projection_angle <= 20:
                        continue
                    if obstructs_in_list([striker_x,STRIKER_Y],modified_coin,our_coins + their_coins):
                        continue
                
                if isBlocked([striker_x,STRIKER_Y], our_coins+their_coins):
                    continue

                
                angle_wrt_base = angle(modified_coin,[striker_x,STRIKER_Y])
                if angle_wrt_base < -45 or angle_wrt_base > 225: 
                    continue


                pd = dist(modified_coin, pocket) 
                sd = dist(modified_coin,[striker_x,STRIKER_Y])
                curr_score = (sd + pd)# * projection_angle

                if(curr_score<min_score):
                    min_score = curr_score
                    striker_pos=striker_x
                    striker_angle=angle_wrt_base
                    pa=projection_angle
                    target_coin = modified_coin
                    target_pocket = pocket
                    if code:
                        # print ('projection_angle', projection_angle, math.sin(projection_angle/2.0))
                        force = (0.5 + math.sin(math.radians(projection_angle/2.0))) * 0.2 * (sd + pd)/BOARD_SIZE
                    else:
                        force = 0.2*(sd + pd)/BOARD_SIZE

    if(striker_pos>-1):
        cut_shot_possible=True   #striker_pos is returned from 0-1
        # print "cutshot possible ", target_coin, target_pocket
    return striker_pos, striker_angle, force


def closest_coin(coins):
    min_y_distance = BOARD_SIZE
    striker_x = -1
    striker_angle = -90
    force = -1
    for coin in coins:

        if abs(coin[1]-STRIKER_Y) <min_y_distance:
            striker_x = min(STRIKER_XR, max(STRIKER_XL, coin[0]))
            striker_angle = angle(coin, [striker_x, STRIKER_Y])
            force = 0.3 * dist(coin, [striker_x, STRIKER_Y])/800

    return striker_x, striker_angle, force


def parse_state_message_heur(msg):
    s = msg.split(";REWARD")
    s[0] = s[0].replace("Vec2d", "")
    reward = float(s[1])
    state = ast.literal_eval(s[0])
    return state, reward


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
learning_rate = 0.07 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength
discount = 0.7


# Initialize the parameters to random values. We need to learn these.
np.random.seed(0)
W1 = np.random.randn(nn_input_dim, nn_output_dim) / np.sqrt(nn_input_dim)
b1 = np.zeros((1, nn_output_dim))
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
    # a1 = np.tanh(z1)        # Activation
    # print(z1, a1)
    # z2 = a1.dot(W2) + b2   # Output Linear layer | 2d matrix
    # a2 = np.tanh(z2) 
    # z3 = a2.dot(W3) + b3
    # print('W2', W2)
    return z1[0][0]  

def backward(error, X):
    global W1, W2, b1, b2
    # dW2 = np.dot(a1.T, error)
    # db2 = np.sum(error, axis=0, keepdims=True)
    # delta2 = error.dot(W2.T) * (1 - np.power(a1, 2))
    dW1 = np.dot(X.T, error)
    db1 = np.sum(error, axis=0)

    # dW3 = (a2.T).dot(error)
    # db3 = np.sum(error, axis=0, keepdims=True)
    # delta3 = error.dot(W3.T) * (1 - np.power(a2, 2))
    # dW2 = np.dot(a1.T, delta3)
    # db2 = np.sum(delta3, axis=0)
    # delta2 = error.dot(W2.T) * (1 - np.power(a1, 2))
    # dW1 = np.dot(a1.T, delta2)
    # db1 = np.sum(delta3, axis=0)
    
    # Add regularization terms (b1 and b2 don't have regularization terms)
    # dW2 += reg_lambda * W2
    dW1 += reg_lambda * W1
    # Gradient descent parameter update
    W1 += -learning_rate * dW1
    b1 += -learning_rate * db1
    # W2 += -learning_rate * dW2
    # b2 += -learning_rate * db2

    
# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent

def gameEnd(state):
    return len(state["White_Locations"]) + len(state["Black_Locations"]) + len(state["Red_Location"]) == 0

Qcurr = 0.0

def agent_1player():
    maxGames = 50
    game = 1

    while game <= maxGames:
        time_step = 1
        coins2 = TWO_PEICE
        state, reward = parse_state_message(coins2, 0)

        maxTime = 500
        # print("Starting game.............", game)
        while time_step <= maxTime: 
            # heuristics ...........
            coins = coins2["White_Locations"] + coins2["Black_Locations"] + coins2["Red_Location"]
            queen = coins2["Red_Location"]
            num_coins = len(coins)
            striker_pos, striker_angle = -1, -1
            striker_force = -1
            queen_cut_shot_exists = False

            #if number of coins is less than 10 and cut shot to queen exists, attempt cut shot to queen
            if len(coins) <= 10 and len(queen) == 1:
                striker_pos, striker_angle, striker_force = cut_shot(coins, [], 1, queen[0])
                if striker_pos > 0:
                    queen_cut_shot_exists = True
            
            #If number of coins is less than 5 and cut shot to queen does not exist, aim at the queen directly
            if not queen_cut_shot_exists and len(coins) <=5 and len(queen) == 1:
                striker_pos, striker_angle, striker_force = cut_shot(coins, [], 0, queen[0])
                queen_cut_shot_exists = True

            # cluster begins here
            if not queen_cut_shot_exists:
                pos, angle, cluster_size, force = select_cluster_1p(coins, 0)
                if(pos < 0):
                    pos, angle, cluster_size, force = select_cluster_1p(coins, 1)
                if(pos >= 0):
                    striker_pos, striker_angle = pos, angle
                    striker_force = force
                    # print "Cluster found close to pocket!"
                else:
                    striker_pos, striker_angle, striker_force = cut_shot(coins, [], 1)
                    if striker_pos < 0:
                        # print "Closest Coin hit"
                        striker_pos, striker_angle, striker_force = cut_shot(coins, [], 0)
                    # else:
                    #     print "cutshot"


            striker_pos=(striker_pos-170.0)/460.0
            striker_angle = (striker_angle +45)/270.0

            # heuristics end ..........


            #  generate action space
            actions = []
            xmin = int((-1) * min(5, (striker_pos/0.05)))
            xmax = int(min(5, ((1-striker_pos)/0.05)))
            anglemin = int((-1) * min(5, striker_angle/0.02))
            anglemax = int(min(5, ((1-striker_angle)/0.02)))
            forcemin = int((-1) * min(5, striker_force/0.02))
            forcemax = int(min(5, ((1-striker_force)/0.02)))
            for x in range(xmin, xmax):
                for angle in range(anglemin, anglemax ):
                    for force in range(forcemin, forcemax):
                        actions.append([striker_pos + 0.05*x, striker_angle + 0.02*angle,  striker_force + 0.02*force])
            nActions = len(actions) 

            # get Q value for every action
            Qs = []
            scores = np.zeros(nActions)
            for a in range(nActions):
                X = np.array(state + actions[a])
                Q = forward(X)
                Qs.append(Q)
                scores[a] = Q
            
            probs = scores / np.sum(scores)
            cdf = [probs[0]]
            for i in range(1, len(probs)):
                cdf.append(cdf[-1] + probs[i])

            a_index_curr = bisect(cdf,random.random())
            Qcurr = Qs[a_index_curr]
            action_picked = actions[a_index_curr]
            angle = -45 + action_picked[1]*270
            action = [action_picked[0], angle, action_picked[2]]
            # print 'action', action

            # simulate
            coins2, reward = one_step.simulate(coins2, one_step.validate(action, coins2))
            if gameEnd(coins2):
                break
            nextState, reward = parse_state_message(coins2, reward)

            if reward <= 0:
                reward = reward + max(-0.25*pow(1.1, time_step), -2)

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
	# try:
	# 	state, reward = parse_state_message(state)  # Get the state and reward
	# except:
	# 	pass

	# coins = state["White_Locations"] + state["Black_Locations"] + state["Red_Location"]
	# if color=="White":
	# 	to_hit = state["White_Locations"] + state["Red_Location"]
	# 	not_to_hit = state["Black_Locations"]
	# else:
	# 	to_hit = state["Black_Locations"] + state["Red_Location"]
	# 	not_to_hit = state["White_Locations"]

	# queen = state["Red_Location"]
	

	# num_coins = len(coins)
	# striker_pos, striker_angle = -1, -1
	# queen_cut_shot_exists = False

 #    #if number of coins is less than 10 and cut shot to queen exists, attempt cut shot to queen
	# if len(to_hit)==2 and len(queen)==1:
	# 	striker_pos, striker_angle, striker_force = cut_shot(to_hit, not_to_hit, 1, queen[0])
	# 	if striker_pos > 0:
	# 		queen_cut_shot_exists = True
	# 	else:
	# 		striker_pos, striker_angle, striker_force = cut_shot(to_hit, not_to_hit, 0, queen[0])
	# 		queen_cut_shot_exists = True

 #    #If number of coins is less than 5 and cut shot to queen does not exist, aim at the queen directly
   

 #    # cluster begins here
	# if not queen_cut_shot_exists:
	# 	striker_pos, striker_angle, striker_force = cut_shot(to_hit, not_to_hit, 1)
	# 	if striker_pos < 0:
	# 		pos, angle, cluster_size, force = select_cluster(to_hit, not_to_hit, 0)
	# 		if(pos < 0):
	# 			pos, angle, cluster_size, force = select_cluster(to_hit, not_to_hit, 1)
				
	# 		if(pos >= 0):
	# 			striker_pos, striker_angle = pos, angle
	# 			striker_force = force
	# 			print "Cluster found close to pocket!"
	# 		else:
	# 			print "Closest Coin hit"
	# 			striker_pos, striker_angle, striker_force = cut_shot(to_hit, not_to_hit, 0)
	# 	else:
	# 		print "Cutshot\n"


	# striker_pos=(striker_pos-170.0)/460.0
	# if timestep == 0 and color=="White":
	# 	striker_pos = 0
	# 	striker_angle = 80
	# 	striker_force = 0.5
		

	# # Can be ignored for now
	# a = str(striker_pos) + ',' + \
	# 	str(striker_angle) + ',' + str(striker_force)
 # 	print a

	# try:
	# 	s.send(a)
	# except Exception as e:
	# 	print "Error in sending:",  a, " : ", e
	# 	print "Closing connection"
	# 	flag = 0

	# return flag

timestep = 0
agent_1player()

