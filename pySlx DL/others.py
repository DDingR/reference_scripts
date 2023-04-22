import numpy as np

def rewardFunction():
    pass


def randomVariables():
    params = {
        'vehicleModel': {
            'X_o': randomRange(-2,2),
            'Y_o': randomRange(-0.1,0.1),
            'ydot_o': randomRange(-0.1,0.1),
            'psi_o': randomRange(-0.01,0.01),
            'r_o': randomRange(-0.01,0.01)
        },
        'Vx': {
            # 'Value': randomRange(25, 35)
            'Value': 30
        },
        'u': {
            'Value': randomRange(-np.pi/12, np.pi/12)
        }
    }
    return params

def randomRange(max, min):
    return (np.random.rand(1,1)*2-1) * (max-min)/2 + (max+min)/2

def rewardCalc(obs, ref):
    err_y = ref[0] - obs[0]
    err_ydot = ref[1] - obs[1]
    err_psi = ref[2] - obs[2]
    err_r = ref[3] - obs[3]
    err_X = obs[6] - obs[4]
    err_Y = obs[7] - obs[5]

    state = [err_y, err_ydot, err_psi, err_r, err_X, err_y]
    dis = np.sqrt(err_X**2 + err_Y**2)

    if dis > 5.0:
        isDone = True
    else:
        isDone = False

    state = np.array(state).reshape(1, len(state))
    reward = np.array((5.0-dis) / 5.0 * 0.1).reshape(1, 1)
    isDone = np.array(isDone).reshape(1,1)

    return state, reward, isDone


def referenceGenerator(max_step, Ts):
    ref = np.ones([max_step , 4])

    ref[:,0] = ref[:,0] * 0
    ref[:,1] = ref[:,1] * 0

    dot_psi = np.pi/6/12

    q = int(max_step/4)

    q_time = np.arange(q) * Ts
    psi = q_time * dot_psi

    ref[0:q,2] = 0
    ref[q:q*2,2] = psi
    ref[q*2:q*3,2] = psi[-1]-psi
    ref[q*3:,2] = 0;

    ref[0:q,3] = 0
    ref[q:q*2,3] = dot_psi
    ref[q*2:q*3,3] = -dot_psi
    ref[q*3:,3] = 0;

    return ref

