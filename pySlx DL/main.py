from simManager import SimManager
from reporter import Reporter, TrainReporter
from others import randomVariables, referenceGenerator, rewardCalc
from agent import PPO_Agent
import argparse
from datetime import datetime
import json

def main(args):
    print("[MAIN] Main Function Called")

    with open(args.param_path) as f:
        param = json.load(f)
    train_param = param['train_param']
    agent_param = param['agent_param']

    modelName = args.slx_name
    DO_TRAIN = args.train
    MAX_EPISODE = train_param['MAX_EPISODE']
    MAX_TIME = train_param['MAX_TIME']
    Ts = train_param['TIME_DIFFERENCE']
    max_steps = int(MAX_TIME / Ts)

    obsInfo = [
        'y', 
        'dot_y',
        'psi',
        'r',
        'X',
        'Y',
        'X_ref',
        'Y_ref',
        'input'
    ]

    reporter = Reporter()
    trainReporter = TrainReporter()
    trainReporter.reportSession(args, train_param, agent_param)

    agent = PPO_Agent(agent_param, args.train_name, args.load_network_path)
    ref = referenceGenerator(max_steps, Ts)

    sim = SimManager(modelName)
    sim.connectMatlab()

    print('[MAIN] Reset Simulation')

    if DO_TRAIN == False:
        print('[MAIN] Demostrain Start')
        MAX_EPISODE = 1
    else:
        print('[MAIN] Train Start')

    for episode in range(MAX_EPISODE):
        step = 0
        initial_param = randomVariables()
        _, obs = sim.reset(obsInfo, initial_param)
        trainReporter.initReward()

        state, _, _ = rewardCalc(obs, ref[0,:])
        while True:
            action = agent.get_action(state)
            abj_parameters = {
                # 'y_ref': {'value': ref[step, 0]},
                'dot_y_ref': {'value': ref[step, 1]},
                'psi_ref': {'value': ref[step, 2]},
                'r_ref': {'value': ref[step, 3]},

                'u': {'value': action}
            }

            time, obs = sim.step(obsInfo, abj_parameters)
            next_state, reward, isDone = rewardCalc(obs, ref[step,:])

            if isDone[0][0] or step == max_steps-1:
                break

            if DO_TRAIN:
                agent.sample_append(state, action, reward, next_state, isDone)
                agent.train()
                agent.draw_tensorboard(reward[0][0], step, episode)

            else:
                reporter.saveRecord(time, obs)

            step += 1
            state = next_state
            trainReporter.saveReward(reward[0][0])

        if DO_TRAIN:
            trainReporter.reportRewardRecord()

            if trainReporter.reward > trainReporter.max_reward:
                trainReporter.max_reward = trainReporter.reward
                
                agent.saveImprovedWeight(now, episode)
      
    if DO_TRAIN:
        print("[MAIN] Train Finished. Final Reward: {}".format(reward))
        print(trainReporter.rewardRecord)
        trainReporter.plotRecord()
    else:
        print("[MAIN] Demostration Finished. Reward: {}".format(round(trainReporter.reward),3))
        reporter.plotRecord(obsInfo)
    print('[MAIN] Simulation Finished')
    sim.disconnectMatlab()

if __name__ == "__main__":
    now = datetime.now()
    now = now.strftime('%m%d%H%M')

    parser = argparse.ArgumentParser(description = "Available Options")
    parser.add_argument('--train_name', type=str,
                        default=now, dest="train_name", action="store",
                        help='trained model would be saved in typed directory')
    parser.add_argument('--slx_name', type=str,
                        default='test_slx', dest='slx_name', action="store",
                        help='simulink model that you want to use as envirnoment')
    parser.add_argument('--train',
                        default=False, dest='train', action='store_true',
                        help='type this arg to train')
    parser.add_argument('--load_network', type=str,
                        default='NOT_LOADED', dest='load_network_path', action='store',
                        help='to load trained network, add path')
    parser.add_argument('--parameters', type=str,
                        default='./train_param.json', dest='param_path', action="store",
                        help='type path of train parameters file (json)')

    args = parser.parse_args()

    main(args)
