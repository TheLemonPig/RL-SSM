import numpy as np
import pytensor
import pytensor.tensor as pt


def rlwm_step(dR, pA, pG, pP, dq_RL, dq_WM):
    cond = pt.switch((dR - dq_RL) >= 0, 1, 0)
    dq_RL = dq_RL + (cond + (1.0 - cond) * pG) * pA * (dR - dq_RL)
    dq_WM = dq_WM + (cond + (1.0 - cond) * pG) * 1.0 * (dR - dq_RL)
    dq_WM = dq_WM + pP * ((1 / dR.shape[2]) - dq_WM)
    return dq_RL, dq_WM


def rlwm_scan(dR, pA, pG, pP, dq_RL, dq_WM):
    results, _ = pytensor.scan(rlwm_step, sequences=[dR, pA, pG, pP], non_sequences=[], outputs_info=[dq_RL, dq_WM])
    dq_RL, dq_WM = results
    return dq_RL, dq_WM


def rlwm_softmax(Qs, pB):
    shape = Qs.shape
    tempered_qs = Qs * pB
    qs_max = tempered_qs.max(axis=2)
    qs_max = pt.repeat(qs_max.reshape((shape[0], shape[1], 1, shape[3], shape[4])), shape[2], axis=2)
    numerator = pt.exp(tempered_qs - qs_max)
    denominator = numerator.sum(axis=2)
    denominator = pt.repeat(denominator.reshape((shape[0], shape[1], 1, shape[3], shape[4])), shape[2], axis=2)
    Ps = numerator / denominator
    return Ps


def rlwm_likelihood(dC, dq_RL, dq_WM, pB, pC, pE, pR, set_sizes):
    weight = pR * pt.clip(pC / set_sizes, 1, pt.inf)
    Ps_RL = rlwm_softmax(dq_RL, pB)
    Ps_WM = rlwm_softmax(dq_WM, pB)
    pol = weight * Ps_WM + (1 - weight) * Ps_RL
    pol_final = (1 - pE) * pol + pE * (1 / dC.shape[2])
    likelihood = pt.log(pol_final)
    return likelihood


def rlwm_recovery(dC, dR, pA, pB, pC, pE, pG, pP, pR, set_sizes):
    dq_RL = pt.ones_like(dC) * 1.0 / dC.shape[2]
    dq_WM = pt.ones_like(dC) * 1.0 / dC.shape[2]
    dq_RL, dq_WM = rlwm_scan(dR, pA, pG, pP, dq_RL, dq_WM)
    likelihood = rlwm_likelihood(dC, dq_RL, dq_WM, pB, pC, pE, pR, set_sizes)
    return likelihood


def rlwm_step_compile():
    dR4 = pt.dtensor4("dR4")
    dq_RL4 = pt.dtensor4("dq_RL4")
    dq_WM4 = pt.dtensor4("dq_WM4")
    pA4 = pt.dtensor4("pA4")
    pG4 = pt.dtensor4("pG4")
    pP4 = pt.dtensor4("pP4")

    dq_RL, dq_WM = rlwm_step(dR4, pA4, pG4, pP4, dq_RL4, dq_WM4)
    rlwm_step_func = pytensor.function(inputs=[dR4, pA4, pG4, pP4, dq_RL4, dq_WM4], outputs=[dq_RL, dq_WM])

    return rlwm_step_func


def rlwm_scan_compile():
    dR5 = pt.dtensor5("dR5")
    dq_RL5 = pt.dtensor5("dq_RL5")
    dq_WM5 = pt.dtensor5("dq_WM5")
    pA5 = pt.dtensor5("pA5")
    pG5 = pt.dtensor5("pG5")
    pP5 = pt.dtensor5("pP5")

    dq_RL, dq_WM = rlwm_scan(dR5, pA5, pG5, pP5, dq_RL5, dq_WM5)
    rlwm_step_func = pytensor.function(inputs=[dR5, pA5, pG5, pP5, dq_RL5, dq_WM5], outputs=[dq_RL, dq_WM])

    return rlwm_step_func


def rlwm_likelihood_compile():
    dC5 = pt.dtensor5("dC5")
    dq_RL5 = pt.dtensor5("dq_RL5")
    dq_WM5 = pt.dtensor5("dq_WM5")
    pB5 = pt.dtensor5("pB5")
    pC5 = pt.dtensor5("pC5")
    pE5 = pt.dtensor5("pE5")
    pR5 = pt.dtensor5("pR5")
    set_sizes = pt.dtensor5("set_sizes")

    likelihood = rlwm_likelihood(dC5, dq_RL5, dq_WM5, pB5, pC5, pE5, pR5, set_sizes)
    rlwm_likelihood_func = pytensor.function(inputs=[dC5, dq_RL5, dq_WM5, pB5, pC5, pE5, pR5, set_sizes], outputs=[likelihood])

    return rlwm_likelihood_func


def rlwm_recovery_compile():
    dC = pt.dtensor5("dC")
    dR = pt.dtensor5("dR")
    pA = pt.dtensor5("pA")
    pG = pt.dtensor5("pG")
    pP = pt.dtensor5("pP")
    pB = pt.dtensor5("pB")
    pC = pt.dtensor5("pC")
    pE = pt.dtensor5("pE")
    pR = pt.dtensor5("pR")
    set_sizes = pt.dtensor5("set_sizes")

    likelihood = rlwm_recovery(dC, dR, pA, pB, pC, pE, pG, pP, pR, set_sizes)
    rlwm_recovery_func = pytensor.function(inputs=[dC, dR, pA, pB, pC, pE, pG, pP, pR, set_sizes], outputs=[likelihood])

    return rlwm_recovery_func


def rlwm_step_test():
    n_participants = 6
    n_choices = 3
    n_stimuli = [4, 5, 6, 7, 8]
    max_stimuli = max(n_stimuli)
    n_blocks = 5
    shape4 = (n_participants, n_choices, max_stimuli, n_blocks)

    dR4_ = np.random.randint(low=0, high=1, size=shape4)
    dq_RL4_ = np.ones_like(dR4_) * 1 / n_choices
    dq_WM4_ = np.ones_like(dR4_) * 1 / n_choices
    pA4_ = np.ones_like(dR4_) * 0.1
    pG4_ = np.ones_like(dR4_) * 0.9
    pP4_ = np.ones_like(dR4_) * 1.0

    test_func = rlwm_step_compile()

    return test_func(dR4_, dq_RL4_, dq_WM4_, pA4_, pG4_, pP4_)


def rlwm_scan_test():
    n_trials = 23
    n_participants = 6
    n_choices = 3
    n_stimuli = [4, 5, 6, 7, 8]
    max_stimuli = max(n_stimuli)
    n_blocks = 5
    shape5 = (n_trials, n_participants, n_choices, max_stimuli, n_blocks)

    dR5_ = np.random.randint(low=0, high=1, size=shape5)
    dq_RL5_ = np.ones_like(dR5_) * 1 / n_choices
    dq_WM5_ = np.ones_like(dR5_) * 1 / n_choices
    pA5_ = np.ones_like(dR5_) * 0.1
    pG5_ = np.ones_like(dR5_) * 0.9
    pP5_ = np.ones_like(dR5_) * 1.0

    test_func = rlwm_scan_compile()

    return test_func(dR5_, dq_RL5_, dq_WM5_, pA5_, pG5_, pP5_)


def rlwm_likelihood_test():
    n_trials = 23
    n_participants = 6
    n_choices = 3
    n_stimuli = [4, 5, 6, 7, 8]
    max_stimuli = max(n_stimuli)
    n_blocks = 5
    shape5 = (n_trials, n_participants, n_choices, max_stimuli, n_blocks)

    dC5_ = np.random.randint(low=0, high=n_choices, size=shape5)
    dq_RL5_ = np.ones_like(dC5_) * 1 / n_choices
    dq_WM5_ = np.ones_like(dC5_) * 1 / n_choices
    pB5_ = np.ones_like(dC5_) * 1.0
    pC5_ = np.ones_like(dC5_) * 4.0
    pE5_ = np.ones_like(dC5_) * 0.5
    pR5_ = np.ones_like(dC5_) * 0.8
    set_sizes_ = np.tile(n_stimuli, (shape5[0], shape5[1], shape5[2], shape5[3], 1))

    test_func = rlwm_likelihood_compile()

    return test_func(dC5_, dq_RL5_, dq_WM5_, pB5_, pC5_, pE5_, pR5_, set_sizes_)


def rlwm_recovery_test():
    n_trials = 23
    n_participants = 6
    n_choices = 3
    n_stimuli = [4, 5, 6, 7, 8]
    max_stimuli = max(n_stimuli)
    n_blocks = 5
    shape = (n_trials, n_participants, n_choices, max_stimuli, n_blocks)

    dR_ = np.random.randint(low=0, high=1, size=shape)
    dC_ = np.random.randint(low=0, high=n_choices, size=shape)
    pA_ = np.ones_like(dC_) * 0.1
    pB_ = np.ones_like(dC_) * 1.0
    pC_ = np.ones_like(dC_) * 4.0
    pE_ = np.ones_like(dC_) * 0.5
    pG_ = np.ones_like(dC_) * 1.0
    pP_ = np.ones_like(dC_) * 1.0
    pR_ = np.ones_like(dC_) * 0.8
    set_sizes_ = np.tile(n_stimuli, (shape[0], shape[1], shape[2], shape[3], 1))

    test_func = rlwm_recovery_compile()

    return test_func(dC_, dR_, pA_, pB_, pC_, pE_, pG_, pP_, pR_, set_sizes_)


if __name__ == "__main__":
    print(rlwm_step_test())