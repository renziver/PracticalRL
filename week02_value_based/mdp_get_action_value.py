
def get_action_value(mdp, state_values, state, action, gamma):
    """ Computes Q(s,a) as in formula above """
    
    # Get next state
    next_states = mdp.get_next_states(state, action)
    
    # Compute action value function
    q_sa = 0
    for s_prime in next_states:
        reward = mdp.get_reward(state, action, s_prime)
        trans_prob = mdp.get_transition_prob(state, action, s_prime)
        value_prime = state_values[s_prime]
        q_sa += trans_prob * (reward + (gamma * value_prime))
    return q_sa
