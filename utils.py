import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns




#### Utility functions regarding the MABC environment

def compute_reward(u0,u1):
    return u0*(1-u1)+u1*(1-u0)



def belief_update(belief,prescription_0,prescription_1,u0,u1,p0,p1):
        
    new_belief = torch.zeros_like(belief)
    N = len(u0)
    
    for i in range(N):
            
        if u0[i]==1 and u1[i]==1:
            new_belief[i,0] = 1.0
            new_belief[i,1] = 1.0
            
        elif u0[i]==0 and u1[i]==1:
                
            if prescription_0[i] == 0:
                new_belief[i,0] = p0*(1-belief[i,0]) + belief[i,0]
            else:
                new_belief[i,0] = p0
                
            new_belief[i,1] = p1
            
        elif u0[i]==1 and u1[i]==0:
                
            new_belief[i,0] = p0
                
            if prescription_1[i] == 0:
                new_belief[i,1] = p1*(1-belief[i,1]) + belief[i,1]
            else:
                new_belief[i,1] = p1
            
        else:
            if prescription_0[i] == 0:
                new_belief[i,0] = p0*(1-belief[i,0]) + belief[i,0]
            else:
                new_belief[i,0] = p0
                
            if prescription_1[i] == 0:
                new_belief[i,1] = p1*(1-belief[i,1]) + belief[i,1]
            else:
                new_belief[i,1] = p1
                
    
    return new_belief



def reset(rollout_size,p0,p1):
    state0 = torch.from_numpy(np.random.binomial(1,p0,rollout_size))
    state1 = torch.from_numpy(np.random.binomial(1,p1,rollout_size))
    belief = torch.cat((torch.tensor([p0]*rollout_size).reshape(-1,1),torch.tensor([p1]*rollout_size).reshape(-1,1)),dim=1)
    
    return state0,state1,belief

def transition(state0,state1,u0,u1,p0,p1):
    rollout_size = len(u0)
    new_state0 = torch.min(state0-u0*(1-u1)+ torch.from_numpy(np.random.binomial(1,p0,rollout_size)),torch.ones_like(state0))
    new_state1 = torch.min(state1-u1*(1-u0)+ torch.from_numpy(np.random.binomial(1,p1,rollout_size)),torch.ones_like(state1))
    
    return new_state0,new_state1

        

    
def compute_returns(rewards,rollout_size,horizon,gamma):
    # Accumulate discounted return
    returns =  torch.zeros(rollout_size, horizon, requires_grad=False)
    temp = torch.zeros(rollout_size,requires_grad=False)
    for step in reversed(range(horizon)):
        #self.returns[:,step] = self.returns[:,step + 1] *gamma + self.rewards[:,step]
        returns[:,step] = temp *gamma + rewards[:,step]
        temp = returns[:,step]
    
    
    return returns



def dummy_func(state0,state1):
    return state0.clone(),state1.clone()




# Computing the reward to go at each time step    
def compute_episode_returns(eps_rewards,rollout_size,horizon,gamma):
    # Accumulate discounted return
    returns =  torch.zeros(rollout_size, horizon, requires_grad=False)
    temp = torch.zeros(rollout_size,requires_grad=False)
    for step in reversed(range(horizon)):
        #self.returns[:,step] = self.returns[:,step + 1] *gamma + self.rewards[:,step]
        returns[:,step] = temp *gamma + eps_rewards[step]
        temp = returns[:,step]
    
    
    return returns







def my_belief_update(belief,prescription_0,prescription_1,u0,u1,p0,p1):
        
    new_belief = torch.zeros_like(belief)
    N = len(u0)
    
    for i in range(N):
            
        lamda0 = (u0[i]==prescription_0[i,1])*belief[i,0]/((u0[i]==prescription_0[i,1])*belief[i,0] +  (u0[i]==prescription_0[i,0])*(1-belief[i,0]))
        lamda1 = (u1[i]==prescription_1[i,1])*belief[i,1]/((u1[i]==prescription_1[i,1])*belief[i,1] +(u1[i]==prescription_1[i,0])*(1-belief[i,1]))
                     
        if u0[i]==0 and u1[i]==1:
            
            new_belief[i,0] = p0*(1-lamda0) + lamda0
            new_belief[i,1] = p1
            
        elif u0[i]==1 and u1[i]==0:
                
            new_belief[i,0] = p0
            new_belief[i,1] = p1*(1-lamda1) + lamda1
            
        else:
            new_belief[i,0] = p0*(1-lamda0) + lamda0
            new_belief[i,1] = p1*(1-lamda1) + lamda1
    
    return new_belief



# State transition function
def my_transition(state0,state1,u0,u1,p0,p1):
    rollout_size = len(u0)
    new_state0 = torch.max(torch.min(state0-u0*(1-u1)+ torch.from_numpy(np.random.binomial(1,p0,rollout_size)),torch.ones_like(state0)),torch.zeros_like(state0))
    new_state1 = torch.max(torch.min(state1-u1*(1-u0)+ torch.from_numpy(np.random.binomial(1,p1,rollout_size)),torch.ones_like(state1)),torch.zeros_like(state0))
    
    return new_state0,new_state1





################## Reward Functions #####################################

def my_reward(x0,x1,u0,u1):
    return x0*u0*(1-u1)+x1*u1*(1-u0)


def diverse_reward(x0,x1,u0,u1):
    L = len(x0)
    reward = torch.zeros_like(x0)
    
    for i in range(L):
        if x0[i]==0 and x1[i]==0:
            reward[i] = 0
        elif x0[i]==0 and x1[i]==1 and u0[i]==0 and u1[i]==1:
            reward[i] = 1.5
        elif x0[i]==0 and x1[i]==1 and u0[i]==0 and u1[i]==0:
            reward[i] = -1
        elif x0[i]==1 and x1[i]==0 and u0[i]==1 and u1[i]==0:
            reward[i] = 1.5
        elif x0[i]==1 and x1[i]==0 and u0[i]==0 and u1[i]==0:
            reward[i] = -1
        elif x0[i]==1 and x1[i]==1 and u0[i]==1 and u1[i]==0:
            reward[i] = 1
        elif x0[i]==1 and x1[i]==1 and u0[i]==0 and u1[i]==1:
            reward[i] = 1
        elif x0[i]==1 and x1[i]==1 and u0[i]==1 and u1[i]==1:
            reward[i] = -1
        else:
            reward[i] = 0
    
    return reward


def penalize_state_reward(x0,x1,u0,u1):
    L = len(x0)
    reward = torch.zeros_like(x0)
    
    for i in range(L):
        if x0[i]==0 and x1[i]==0:
            reward[i] = 1
        else:
            reward[i] = 0
    
    return reward



def success_with_buffer(x0,x1,u0,u1):
    return (x0>0)*u0*(1-u1)+(x1>0)*u1*(1-u0)
############################## Plotting utilities ##############################################

def plot_logp(policy_0,policy_1,zero_enforced=False):
    
    probs = [0.0,0.3,0.5,0.8,1.0]
    if zero_enforced:
        
        L = len(probs)
        log_p0 = np.zeros(shape=(L,L))
        log_p1 = np.zeros(shape=(L,L))
        
        
        for i1,b0 in enumerate(probs):
            for i2,b1 in enumerate(probs):
                inp_state = torch.tensor([b0,b1]).reshape(1,2)
                log0 = policy_0.fc(inp_state)
                log1 = policy_1.fc(inp_state)
                log_p0[i1,i2] = log0.data.numpy().squeeze()[0]
                log_p1[i1,i2] = log1.data.numpy().squeeze()[0]
        
       
        f,axes = plt.subplots(1,2)
        plt.hold(True)
        for i in range(L):
            axes[0].plot(probs,log_p0[:,i],label='b1={}'.format(probs[i]))
            axes[1].plot(probs,log_p1[i,:],label='b0={}'.format(probs[i]))
                
        axes[0].legend()
        axes[0].set_title('log_prob vs belief of agent 0 for u = 0')
        axes[1].legend()
        axes[1].set_title('log_prob vs belilef of agent 1 for u = 0')
        
        plt.gcf().set_size_inches(10, 12)
        plt.subplots_adjust(hspace=0.5)
        plt.show()
    else:
        L = len(probs)
        log_p0 = np.zeros(shape=(L,L,2,2))
        log_p1 = np.zeros(shape=(L,L,2,2))
        
        
        for i1,b0 in enumerate(probs):
            for i2,b1 in enumerate(probs):
                for j in range(2):
                    inp_state = torch.tensor([b0,b1,j*1.0]).reshape(1,3)
                    log0 = policy_0.fc(inp_state)
                    log1 = policy_1.fc(inp_state)
                    log_p0[i1,i2,j,:] = log0.data.numpy().squeeze()
                    log_p1[i1,i2,j,:] = log1.data.numpy().squeeze()
        
       
        f,axes = plt.subplots(4,2)
        plt.hold(True)
        for x in range(2):
            for u in range(2):
                for i in range(L):
                    axes[2*x+u,0].plot(probs,log_p0[:,i,x,u],label='b1={}'.format(probs[i]))
                    axes[2*x+u,1].plot(probs,log_p1[i,:,x,u],label='b0={}'.format(probs[i]))
                
                axes[2*x+u,0].legend()
                axes[2*x+u,0].set_title('log_p0 vs b0 for (x,u) ={},{}'.format(x,u))
                axes[2*x+u,1].legend()
                axes[2*x+u,1].set_title('log_p1 vs b1 for (x,u) ={},{}'.format(x,u))
        
        plt.gcf().set_size_inches(10, 12)
        plt.subplots_adjust(hspace=0.5)
        plt.show()
        

        

        

        
def plot_reward(rewards,plot_std=True):
    
    r = np.asarray(rewards)
    df = pd.DataFrame(r).melt()
    sns.lineplot(x='variable',y='value',data=df,label='No. of success per episode')
    plt.xlabel('Training iterations')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid('on')
    plt.show()
    
    


############### Utility functions for MABC problem with buffer size larger than 1 ####################################    



def belief_update_control_sharing(belief,prescription_0,prescription_1,u0,u1,transition_prob):
    """
    This function implements the general belief update for one step delayed control sharing information structure
    
    belief : current belief -- |X|^2 long vector
    prescription_i : prescription to agent i  -- |X| long vector for each i
    ui : Control action of agent i
    transition_prob - Transition probabilities -- |X|^2 \times |X|^2 \times |U|^2 tensor
    
    """
    
    N,state_size = prescription_0.shape
    
    
    presc_0_transpose = prescription_0.T
    presc_1_transpose = prescription_1.T
    
    new_belief = torch.zeros_like(belief.T)
    
    _,_,U_square = transition_prob.shape
    control_size = int(np.sqrt(U_square))
    
    #prob_matrix = transition_prob[:,:,u0*control_size+u1]
    
    I0 = presc_0_transpose==u0
    I1 = presc_1_transpose==u1
    
    Ind_matrix = I0[0,:]*I1
    
    for i in range(1,state_size):
        Ind_matrix = torch.cat([Ind_matrix,I0[i,:]*I1],dim=0)
    
    belief_times_ind = Ind_matrix*belief.T
    belief_times_ind_sum = torch.sum(belief_times_ind,axis=0)
    
    for i in range(N):
        new_belief[:,i] = torch.matmul(transition_prob[:,:,u0[i]*control_size+u1[i]],belief_times_ind[:,i])/belief_times_ind_sum[i]
    
    return new_belief.T


def transition_with_buffer(state0,state1,u0,u1,p0,p1,buffer_size):
    rollout_size = len(u0)
    
    new_state0 = torch.max(torch.min(state0-u0*(1-u1)+ torch.from_numpy(np.random.binomial(1,p0,rollout_size)),buffer_size*torch.ones_like(state0)),torch.zeros_like(state0))
    
    new_state1 = torch.max(torch.min(state1-u1*(1-u0)+ torch.from_numpy(np.random.binomial(1,p1,rollout_size)),buffer_size*torch.ones_like(state1)),torch.zeros_like(state0))
    
    return new_state0,new_state1



def reset_with_buffer(rollout_size,p0,p1,buffer_size):
    state0 = torch.from_numpy(np.random.binomial(1,p0,rollout_size))
    state1 = torch.from_numpy(np.random.binomial(1,p1,rollout_size))
    
    #belief = torch.cat((torch.tensor([p0]*rollout_size).reshape(-1,1),torch.tensor([p1]*rollout_size).reshape(-1,1)),dim=1)
    a = torch.tensor([1-p0,p0])
    b = torch.zeros(buffer_size-1)
    belief0 = torch.cat((a,b))
    
    a = torch.tensor([1-p1,p1])
    belief1 = torch.cat((a,b))
    
    joint_belief = torch.zeros((buffer_size+1)**2)
    
    for i0 in range(buffer_size+1):
        for i1 in range(buffer_size+1):
            joint_belief[i0*(buffer_size+1)+i1] = belief0[i0]*belief1[i1]
    
    joint_belief = (joint_belief.repeat(rollout_size,1))
    
    return state0,state1,joint_belief



def sample(prob,N):
    """
    Generates N samples from the probability distribution prob
    """
    samples = np.random.choice(len(prob),N,p=prob)
    return torch.from_numpy(samples)
        
    

def generate_transition_prob(p0,p1,buffer_size):
    
    buffer_size = buffer_size + 1 # state space is one larger than buffer_size
    
    transition_prob_0 = np.zeros(shape=(buffer_size,buffer_size,4))
    transition_prob_1 = np.zeros(shape=(buffer_size,buffer_size,4))
    
    for u0 in range(2):
        for u1 in range(2):
            
            if u0^u1==0:
                
                for j in range(buffer_size-1):
                    
                    transition_prob_0[j,j,2*u0+u1] = (1-p0)
                    transition_prob_0[j+1,j,2*u0+u1] = p0
                    
                    transition_prob_1[j,j,2*u0+u1] = (1-p1)
                    transition_prob_1[j+1,j,2*u0+u1] = p1
                
                transition_prob_0[buffer_size-1,buffer_size-1,2*u0+u1] = 1.0
                transition_prob_1[buffer_size-1,buffer_size-1,2*u0+u1] = 1.0
               
            elif u0==1:
                
                for j in range(buffer_size-1):
                    
                    if j==0:
                        transition_prob_0[0,0,2*u0+u1] = 1-p0
                        transition_prob_0[1,0,2*u0+u1] = p0
                    else:
                        transition_prob_0[j-1,j,2*u0+u1] = 1-p0
                        transition_prob_0[j,j,2*u0+u1] = p0
                    
                    transition_prob_1[j,j,2*u0+u1] = (1-p1)
                    transition_prob_1[j+1,j,2*u0+u1] = p1
                    
                
                transition_prob_0[buffer_size-1,buffer_size-1,2*u0+u1] = p0
                transition_prob_0[buffer_size-2,buffer_size-1,2*u0+u1] = 1-p0
                transition_prob_1[buffer_size-1,buffer_size-1,2*u0+u1] = 1.0
                
            else:
                
                for j in range(buffer_size-1):
                    
                    if j==0:
                        transition_prob_1[0,0,2*u0+u1] = 1-p1
                        transition_prob_1[1,0,2*u0+u1] = p1
                    else:
                        transition_prob_1[j-1,j,2*u0+u1] = 1-p1
                        transition_prob_1[j,j,2*u0+u1] = p1
                    
                    transition_prob_0[j,j,2*u0+u1] = (1-p0)
                    transition_prob_0[j+1,j,2*u0+u1] = p0
                    
                
                transition_prob_1[buffer_size-1,buffer_size-1,2*u0+u1] = p1
                transition_prob_1[buffer_size-2,buffer_size-1,2*u0+u1] = 1-p1
                transition_prob_0[buffer_size-1,buffer_size-1,2*u0+u1] = 1.0
                
                
    
    joint_transition_prob = np.zeros(shape=(buffer_size**2,buffer_size**2,4))
    
    for u0 in range(2):
        for u1 in range(2):
            for i1 in range(buffer_size):
                for j1 in range(buffer_size):
                    for i2 in range(buffer_size):
                        for j2 in range(buffer_size):
                    
                            joint_transition_prob[buffer_size*i2+j2,buffer_size*i1+j1,2*u0+u1] =transition_prob_0[i2,i1,2*u0+u1]*transition_prob_1[j2,j1,2*u0+u1]
                        
                

    return joint_transition_prob

    

    
