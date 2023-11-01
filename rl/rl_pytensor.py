import numpy as np
import pytensor
import pytensor.tensor as pt


def pt_model():
    # parameters & variables
    C = pt.ivector("C")  # choice vector
    R = pt.ivector("R")  # reward vector
    Q = pt.vector("Q")  # empty matrix to store q-values across trials
    a = pt.scalar("a")  # learning rate

    # function for a single RL step
    def rl_step(c, r, q_tm1):
        m = pt.set_subtensor(pt.zeros_like(q_tm1)[c], pt.constant(1))
        rm = r * m
        qm = q_tm1 * m
        return qm + a * (rm - qm)

    # scan function build
    Qs, updates = pytensor.scan(rl_step, sequences=[C, R], outputs_info=Q)

    # Q Prediction function compilation
    return pytensor.function(inputs=[C, R, Q, a], outputs=Qs)


if __name__ == "__main__":
    # test values
    C_test = np.ones((100,), dtype=np.int32)
    R_test = np.ones((100,), dtype=np.int32)
    Q_test = np.zeros(shape=(2,), dtype=pytensor.config.floatX)
    a_test = 0.1

    # RL Model function test
    q_predict = pt_model()
    pred = q_predict(C_test, R_test, Q_test, a_test)
    print(pred)