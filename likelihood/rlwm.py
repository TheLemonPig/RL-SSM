import numpy as np
import pytensor
import pytensor.tensor as pt


def rlwm_step(dR, dq_RL, dq_WM, pA, pG, pP):
    cond = pt.switch((dR - dq_RL) >= 0, 1, 0)
    dq_RL = dq_RL + (cond + (1.0 - cond) * pG) * pA * (dR - dq_RL)
    dq_WM = dq_WM + (cond + (1.0 - cond) * pG) * 1.0 * (dR - dq_RL)
    dq_WM = dq_WM + pP * ((1 / dR.shape[2]) - dq_WM)
    return dq_RL, dq_WM

def rlwm_likelihood():
    weight = rho * min(1, C / set_size)


        pol_RL = softmax(q_RL[state, :], beta)
        pol_WM = softmax(q_WM[state, :], beta)
        pol = weight * pol_WM + (1 - weight) * pol_RL
        pol_final = (1 - epsilon) * pol + epsilon * np.tile([1 / num_actions], num_actions)
        action = int(action_list[tr])
        ### CHECK -- reward is always 1 (when rew=1 or rew=2) or 0
        if reward_list[tr] == 1 or reward_list[tr] == 2:
            reward = 1
        elif reward_list[tr] == 0:
            reward = 0
        # reward = reward_list[tr]
        subj_ll += np.log(pol_final[action])



def rlwm_step_compile():
    dR4 = pt.dtensor4("dR4")
    pA4 = pt.dtensor4("pA4")
    pG4 = pt.dtensor4("pG4")

def rlwm_step_test():
    dR4_ =
    pA4_ =
    pG4_ =
    dq_RL = np.ones((n_stimuli, n_choices)) * 1 / n_choices
    dq_WM = np.ones((n_stimuli, n_choices)) * 1 / n_choices


if __name__ == "__main__":
    .