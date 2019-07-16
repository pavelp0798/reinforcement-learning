import numpy as np
import matplotlib.pyplot as plt
from degree_freedom_queen import *
from degree_freedom_king1 import *
from degree_freedom_king2 import *
from features import *
from generate_game import *
from Q_values import *
import numpy.matlib
size_board = 4
import random

def main(algorithm):
    """
    Generate a new game
    The function below generates a new chess board with King, Queen and Enemy King pieces randomly assigned so that they
    do not cause any threats to each other.
    s: a size_board x size_board matrix filled with zeros and three numbers:
    1 = location of the King
    2 = location of the Queen
    3 = location fo the Enemy King
    p_k2: 1x2 vector specifying the location of the Enemy King, the first number represents the row and the second
    number the column
    p_k1: same as p_k2 but for the King
    p_q1: same as p_k2 but for the Queen
    """
    s, p_k2, p_k1, p_q1 = generate_game(size_board)
    # print("matrix ->",s, "p_k2 ->",p_k2, "p_k1 ->", p_k1, "p_q1 ->",p_q1)
    """
    Possible actions for the Queen are the eight directions (down, up, right, left, up-right, down-left, up-left, 
    down-right) multiplied by the number of squares that the Queen can cover in one movement which equals the size of 
    the board - 1
    """
    possible_queen_a = (s.shape[0] - 1) * 8
    """
    Possible actions for the King are the eight directions (down, up, right, left, up-right, down-left, up-left, 
    down-right)
    """
    possible_king_a = 8

    # Total number of actions for Player 1 = actions of King + actions of Queen
    N_a = possible_king_a + possible_queen_a

    """
    Possible actions of the King
    This functions returns the locations in the chessboard that the King can go
    dfK1: a size_board x size_board matrix filled with 0 and 1.
          1 = locations that the king can move to
    a_k1: a 8x1 vector specifying the allowed actions for the King (marked with 1): 
          down, up, right, left, down-right, down-left, up-right, up-left
    """
    dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
    # print("1 = locations that the king can move to ->",dfK1,"|", "a_k1: a 8x1 vector specifying the allowed actions for the King ->", a_k1)
    """
    Possible actions of the Queen
    Same as the above function but for the Queen. Here we have 8*(size_board-1) possible actions as explained above
    """
    dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
    """
    Possible actions of the Enemy King
    Same as the above function but for the Enemy King. Here we have 8 possible actions as explained above
    """
    dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)

    """
    Compute the features
    x is a Nx1 vector computing a number of input features based on which the network should adapt its weights  
    with board size of 4x4 this N=50
    """
    x = features(p_q1, p_k1, p_k2, dfK2, s, check)

    n_input_layer = 50  # Number of neurons of the input layer. Moves enemy king can make, checked
    n_hidden_layer = 200  # Number of neurons of the hidden layer
    n_output_layer = 32  # Number of neurons of the output layer. 

    W1 = np.random.uniform(0, 1, (n_hidden_layer, n_input_layer))
    W1 = np.divide(W1, np.matlib.repmat(np.sum(W1,1)[:, None], 1, n_input_layer))

    W2 = np.random.uniform(0,1,(n_output_layer,n_hidden_layer))
    W2 = np.divide(W2,np.matlib.repmat(np.sum(W2,1)[:,None],1,n_hidden_layer))
    # print(W1, W2)
    bias_W1 = np.ones((n_hidden_layer,))
    bias_W2 = np.ones((n_output_layer,))

    # Network Parameters
    epsilon_0 = 0.2   #epsilon for the e-greedy policy
    beta = 0.00005    #epsilon discount factor
    gamma = 0.85      #SARSA Learning discount factor
    eta = 0.0035      #learning rate
    N_episodes = 100000 #Number of games, each game ends when we have a checkmate or a draw
    alpha = 1/10000
    if algorithm == "sarsa":
        sarsa = 1
        qlearning = 0
    else:
        sarsa = 0
        qlearning = 1
    ###  Training Loop  ###

    # Directions: down, up, right, left, down-right, down-left, up-right, up-left
    # Each row specifies a direction, 
    # e.g. for down we need to add +1 to the current row and +0 to current column
    map = np.array([[1, 0],
                    [-1, 0],
                    [0, 1],
                    [0, -1],
                    [1, 1],
                    [1, -1],
                    [-1, 1],
                    [-1, -1]])
    
    R_save = np.zeros([N_episodes, 1])
    N_moves_save = np.zeros([N_episodes, 1])
    
    if_Q_next = 0
    
    for n in range(N_episodes):    
        epsilon_f = epsilon_0 / (1 + beta * n) #psilon is discounting per iteration to have less probability to explore
        checkmate = 0  # 0 = not a checkmate, 1 = checkmate
        draw = 0  # 0 = not a draw, 1 = draw
        i = 1  # counter for movements
        # Generate a new game
        s, p_k2, p_k1, p_q1 = generate_game(size_board)

        # Possible actions of the King
        dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
        # Possible actions of the Queen
        dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
        # Possible actions of the enemy king
        dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)

        while checkmate == 0 and draw == 0:
            
            # Player 1

            # Actions & allowed_actions
            a = np.concatenate([np.array(a_q1), np.array(a_k1)])
            allowed_a = np.where(a > 0)[0]

            # Computing Features
            x = features(p_q1, p_k1, p_k2, dfK2, s, check)

            Q, out1 = Q_values(x, W1, W2, bias_W1, bias_W2)
            
            if np.unique(Q).size == 1 or (int(np.random.rand() < epsilon_f)):
                a_agent = random.choice(allowed_a)
            else:
                Q2 = Q
                done = 0
                while done == 0:
                    move = Q2.argmax()
                    if move in allowed_a:
                        a_agent = move
                        done = 1
                    else:
                        Q2[move] = 0
            

            picked_action = [0]*32
            picked_action[a_agent] = 1

            # Player 1 makes the action
            if a_agent < possible_queen_a:
                direction = int(np.ceil((a_agent + 1) / (size_board - 1))) - 1
                steps = a_agent - direction * (size_board - 1) + 1

                s[p_q1[0], p_q1[1]] = 0
                mov = map[direction, :] * steps
                s[p_q1[0] + mov[0], p_q1[1] + mov[1]] = 2
                p_q1[0] = p_q1[0] + mov[0]
                p_q1[1] = p_q1[1] + mov[1]

            else:
                direction = a_agent - possible_queen_a
                steps = 1

                s[p_k1[0], p_k1[1]] = 0
                mov = map[direction, :] * steps
                s[p_k1[0] + mov[0], p_k1[1] + mov[1]] = 1
                p_k1[0] = p_k1[0] + mov[0]
                p_k1[1] = p_k1[1] + mov[1]

            # Compute the allowed actions for the new position

            # Possible actions of the King
            dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
            # Possible actions of the Queen
            dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
            # Possible actions of the enemy king
            dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)

            # Player 2
            # Check for draw or checkmate
            if np.sum(dfK2) == 0 and dfQ1_[p_k2[0], p_k2[1]] == 1:
                # King 2 has no freedom and it is checked
                # Checkmate and collect reward
                checkmate = 1
                R = 1 # Reward is 1 when checkmate
    
                if if_Q_next == True:
                    x = x.reshape(1, -1)
                    out1 = out1.reshape(1, -1)
                    Q = Q.reshape(1, -1)

                    if sarsa == False:
                        target = R + gamma*max(Q_next)
                    else:
                        target = R
                        
                    di = (target - Q) * picked_action
                    dj = np.dot(di, W2)

                    W1 += (eta * np.dot(x.T,  np.dot(di, W2))).T
                    W2 += (eta * np.dot(out1.T, di)).T
                    bias_W1 += eta *  np.dot(di, W2)[0]
                    bias_W2 += eta * di[0]
                    
                if checkmate:
                    break

            elif np.sum(dfK2) == 0 and dfQ1_[p_k2[0], p_k2[1]] == 0:
                # King 2 has no freedom but it is not checked
                draw = 1
                R = 0.1

                if if_Q_next == True:
                    x = x.reshape(1, -1)
                    out1 = out1.reshape(1, -1)
                    Q = Q.reshape(1, -1)
                    if sarsa == False:
                        target = R + gamma*max(Q_next)
                    else:
                        target = R
                    
                    di = (target - Q) * picked_action
                    dj = np.dot(di, W2)

                    W1 += (eta * np.dot(x.T,  np.dot(di, W2))).T
                    W2 += (eta * np.dot(out1.T, di)).T
                    bias_W1 += eta *  np.dot(di, W2)[0]
                    bias_W2 += eta * di[0]

                if draw:
                    break
            else:
                # Move enemy King randomly to a safe location
                allowed_enemy_a = np.where(a_k2 > 0)[0]
                a_help = int(np.ceil(np.random.rand() * allowed_enemy_a.shape[0]) - 1)
                a_enemy = allowed_enemy_a[a_help]

                direction = a_enemy
                steps = 1

                s[p_k2[0], p_k2[1]] = 0
                mov = map[direction, :] * steps
                s[p_k2[0] + mov[0], p_k2[1] + mov[1]] = 3

                p_k2[0] = p_k2[0] + mov[0]
                p_k2[1] = p_k2[1] + mov[1]

            # Possible actions of the King
            dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
            # Possible actions of the Queen
            dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
            # Possible actions of the enemy king
            dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)
            # Compute features
            x_next = features(p_q1, p_k1, p_k2, dfK2, s, check)
            # Compute Q-values for the discounted factor
            Q_next, _ = Q_values(x_next, W1, W2, bias_W1, bias_W2)

            if_Q_next = True

            if not check or draw:
                x = x.reshape(1, -1)
                out1 = out1.reshape(1, -1)
                Q = Q.reshape(1, -1)
                if sarsa == False:
                    target = R + gamma*max(Q_next)
                else:
                    target = R + gamma*Q_next
                
                di = (target - Q) * picked_action
                dj = np.dot(di, W2)

                W1 += (eta * np.dot(x.T,  np.dot(di, W2))).T
                W2 += (eta * np.dot(out1.T, di)).T
                bias_W1 += eta *  np.dot(di, W2)[0]
                bias_W2 += eta * di[0]

            i += 1
        R_save[n, : ] = ((1-alpha)*R_save[n-1,:]) + (alpha*R)
        N_moves_save[n, : ] = ((1-alpha)*N_moves_save[n-1,:]) + (alpha*i)        
    return N_moves_save, R_save

def displayPlots():
    N_moves_save, R_save = main("sarsa")
    N_moves_save2, R_save2 = main("qlearning")
    fig, ax = plt.subplots(1,2, figsize=(15,15))

    ax[0].plot(N_moves_save, 'b')
    ax[0].plot(N_moves_save2, 'g')
    ax[0].set_xlabel('Number of episodes')
    ax[0].set_ylabel('Moves')
    ax[0].title.set_text('Q-Learning Moves')

    ax[1].plot(R_save, 'b') 
    ax[1].plot(R_save2, 'g') 
    ax[1].set_xlabel('Number of episodes')
    ax[1].set_ylabel('Exponential Moving Average of Reward')
    ax[1].title.set_text('Q-Learning Reward')

    plt.show()
    
if __name__ == '__main__':
    displayPlots()
    