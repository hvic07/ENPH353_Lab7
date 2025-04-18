#!/usr/bin/env python3
import gym
from gym import wrappers
import gym_gazebo
import time
import numpy
import random
import time

import qlearn
import liveplot
import os

from matplotlib import pyplot as plt


def render():
    render_skip = 0  # Skip first X episodes.
    render_interval = 50  # Show render Every Y episodes.
    render_episodes = 10  # Show Z episodes every rendering.

    if (x % render_interval == 0) and (x != 0) and (x > render_skip):
        env.render()
    elif (((x-render_episodes) % render_interval == 0) and (x != 0) and
          (x > render_skip) and (render_episodes < x)):
        env.render(close=True)


if __name__ == '__main__':

    env = gym.make('Gazebo_linefollow-v0')

    outdir = '/tmp/gazebo_gym_experiments'
    env = gym.wrappers.Monitor(env, outdir, force=True)
    plotter = liveplot.LivePlot(outdir)

    last_time_steps = numpy.ndarray(0)
    best_q_table_file = "best_q_values.pkl"


    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                           alpha=0.25, gamma=0.85, epsilon=0.8)
    

    if os.path.exists(best_q_table_file):
        qlearn.loadQ(best_q_table_file)
        print("Loaded previous best Q-table.")
    else:
        print("No saved Q-table found, starting fresh.")

    # qlearn.loadQ("QValues_A+")

    initial_epsilon = qlearn.epsilon

    epsilon_discount = 0.9986#0.9986

    start_time = time.time()
    total_episodes = 10000
    highest_reward = 0

    for x in range(total_episodes):
        done = False

        cumulated_reward = 0  # Should going forward give more reward then L/R?

        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount

        observation = env.reset()
        state = ','.join(map(str, observation))

        # render() #defined above, not env.render()

        i = -1
        while True:
            i += 1

            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)
            nextState = ','.join(map(str, observation))

            qlearn.learn(state, action, reward, nextState)

            cumulated_reward += reward

            env._flush(force=True)

            if not(done):
                state = nextState
            else:
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break

        print("===== Completed episode {}".format(x))

        if (x > 0) and (x % 5 == 0):
            #qlearn.saveQ("QValues")
            plotter.plot(env)

        if cumulated_reward > highest_reward:  # New best reward achieved
            highest_reward = cumulated_reward
            qlearn.saveQ(best_q_table_file)  # Save as "best policy"
            print(f"New best policy saved with reward: {highest_reward}")

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        print ("Starting EP: " + str(x+1) +
               " - [alpha: " + str(round(qlearn.alpha, 2)) +
               " - gamma: " + str(round(qlearn.gamma, 2)) +
               " - epsilon: " + str(round(qlearn.epsilon, 2)) +
               "] - Reward: " + str(cumulated_reward) +
               "     Time: %d:%02d:%02d" % (h, m, s))

    # Github table content
    print ("\n|"+str(total_episodes)+"|"+str(qlearn.alpha)+"|" +
           str(qlearn.gamma)+"|"+str(initial_epsilon)+"*" +
           str(epsilon_discount)+"|"+str(highest_reward) + "| PICTURE |")

    l = last_time_steps.tolist()
    l.sort()

    # print("Parameters: a="+str)
    print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    print("Best 100 score: {:0.2f}".
          format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()