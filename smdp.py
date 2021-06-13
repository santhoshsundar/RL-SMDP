from rooms.env.fourroomsenv import FourRoomsEnv
import numpy as np
from copy import deepcopy
import seaborn as sb
import matplotlib.pyplot as plt
import math
import random
import gym
import rooms

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

EPS = 0.9
n_iterations = 500
n_episode_length = 1000
n_states = 104
n_rooms = 4
n_actions = 4
goal_states = [56, 64]
alpha = 1 / 8
gamma = 0.9

goal1 = [1, [6, 2]]
goal2 = [2, [1, 2]]

specific_start_state = 90
specific_start_state_bool = False
beta_options = []
states_room = []

room_sizes = [[5, 5], [6, 5], [4, 5], [5, 5]]
rooms_options = [
    [0, 1, 8, 9, 10, 11],
    [2, 3, 8, 9, 10, 11],
    [4, 5, 8, 9, 10, 11],
    [6, 7, 8, 9, 10, 11],
]
rooms_states = [[0, 1], [2, 3], [4, 5], [6, 7]]

env = gym.make("Rooms-v0")

p1 = [1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0]
p2 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 3, 3, 3, 3]
p3 = [
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    1,
    2,
    3,
    2,
    1,
    1,
    2,
    3,
    3,
    1,
]
p4 = [
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    3,
    3,
    3,
    3,
    3,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]
p5 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 2]
p6 = [1, 1, 0, 3, 3, 0, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
p7 = [1, 0, 3, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]
p8 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2]

policies = [p1, p2, p3, p4, p5, p6, p7, p8]


def create_offsets(n_rooms, room_sizes):

    offsets = [0] * (n_rooms + 1)
    for i in range(n_rooms):
        offsets[i + 1] = offsets[i] + room_sizes[i][0] * room_sizes[i][1] + 1

    return offsets


def create_qvalues(n_states, goal_state):
    smdp_q = []
    for it in np.arange(n_states):
        l = []
        if it == goal_state:
            for _ in np.arange(6):
                l.append(1)
        else:
            for _ in np.arange(6):
                l.append(0)
        smdp_q.append(l)
    return smdp_q


# Returns Initiation set
def create_states_rooms(offsets):

    index = 0
    states_room = []
    states_room_list = []
    hallway_coord = [
        [103],
        [25],
        [25],
        [56],
        [56],
        [77],
        [77],
        [103],
    ]
    while index <= 3:
        state_room = []
        for i in np.arange(offsets[index], offsets[index + 1] - 1):
            state_room.append(i)
        index = index + 1
        states_room.append(state_room)

    for i in np.arange(n_rooms):
        state_room1 = deepcopy(states_room[i])
        state_room2 = deepcopy(states_room[i])

        state_room1 = state_room1 + hallway_coord[2 * i]
        state_room2 = state_room2 + hallway_coord[(2 * i) + 1]

        states_room_list.append(state_room1)
        states_room_list.append(state_room2)

    return states_room_list


# Returns Beta for every option
def create_terminating_condition_list(n_rooms, beta_options, offsets):

    index = 0
    hallway_coord = [25, 56, 77, 103]
    clk_next_hallway_coord = [36, 59, 97, 21]
    aclk_next_hallway_coord = [14, 53, 67, 79]
    for i in np.arange(n_rooms):
        beta = {}
        for k in np.arange(offsets[index], offsets[index + 1] - 1):
            beta[k] = 0

        beta1 = deepcopy(beta)
        beta2 = deepcopy(beta)

        if i == 0:
            beta1[hallway_coord[3]] = 0
            beta1[hallway_coord[0]] = 1

            beta2[hallway_coord[0]] = 0
            beta2[hallway_coord[3]] = 1

            beta1[clk_next_hallway_coord[0]] = 1
            beta1[aclk_next_hallway_coord[3]] = 1

            beta2[clk_next_hallway_coord[0]] = 1
            beta2[aclk_next_hallway_coord[3]] = 1

        elif i == 1:
            beta1[hallway_coord[1]] = 1
            beta1[hallway_coord[0]] = 0

            beta2[hallway_coord[0]] = 1
            beta2[hallway_coord[1]] = 0

            beta1[clk_next_hallway_coord[1]] = 1
            beta1[aclk_next_hallway_coord[0]] = 1

            beta2[clk_next_hallway_coord[1]] = 1
            beta2[aclk_next_hallway_coord[0]] = 1

        elif i == 2:
            beta1[hallway_coord[2]] = 1
            beta1[hallway_coord[1]] = 0

            beta2[hallway_coord[2]] = 0
            beta2[hallway_coord[1]] = 1

            beta1[clk_next_hallway_coord[2]] = 1
            beta1[aclk_next_hallway_coord[1]] = 1

            beta2[clk_next_hallway_coord[2]] = 1
            beta2[aclk_next_hallway_coord[1]] = 1

        elif i == 3:
            beta1[hallway_coord[3]] = 1
            beta1[hallway_coord[2]] = 0

            beta2[hallway_coord[3]] = 0
            beta2[hallway_coord[2]] = 1

            beta1[clk_next_hallway_coord[3]] = 1
            beta1[aclk_next_hallway_coord[2]] = 1

            beta2[clk_next_hallway_coord[3]] = 1
            beta2[aclk_next_hallway_coord[2]] = 1

        beta_options.append(beta1)
        beta_options.append(beta2)
        index = index + 1

    return beta_options


# Returns options with components<Initiation set,Policies,Beta,Target>
def create_options(states_room, beta_dict, policy_dict, rooms_options, rooms_states):
    options = {}
    index = 0
    hallway_coord = [
        25,
        103,
        56,
        25,
        77,
        56,
        103,
        77,
    ]
    hallway_action = [1, 2, 2, 3, 3, 0, 0, 1]
    opt = ["o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8"]
    for state_room_, policy_, beta_ in zip(states_room, policy_dict, beta_dict):
        option1 = {}
        option1["states"] = state_room_
        option1["policy"] = policy_
        option1["beta"] = beta_
        option1["target"] = hallway_coord[index]
        option1["t_action"] = hallway_action[index]

        options[opt[index]] = option1
        index = index + 1

    return options


def calculate_state_option_reward(gamma, curr_state_timestep, curr_state_reward):
    total_reward = 0
    for step in np.arange(curr_state_timestep):
        total_reward = total_reward + math.pow(gamma, step) * curr_state_reward[step]

    return total_reward


def get_max_q_values(smdp_q):
    qvalue = []
    for state in smdp_q:
        qvalue.append(np.max(state))
    return qvalue


def create_policy_list(states_room, policies):
    policy_list = []
    for i in np.arange(len(policies)):
        pol = {}
        for state, action in zip(states_room[i], policies[i]):
            pol[state] = action
        policy_list.append(pol)
    return policy_list


def max_primitive_action(state, smdp_q):

    c_state_values = smdp_q[state][2:]
    max_value = np.max(c_state_values)

    max_index = [
        index for index, value in enumerate(c_state_values) if value == max_value
    ]

    if len(max_index) == 1:
        action_ = max_index[0]
    else:
        action_ = max_index[random.randint(0, len(max_index) - 1)]

    return action_


def max_value(state, smdp_q_):

    c_state_values = smdp_q_[state]
    max_value = np.max(c_state_values)
    return max_value


def epsilon_greedy_policy(c_action, _EPS):
    rand = 0
    choice_list = []
    rand = np.random.uniform()
    choice_list = list(np.arange(0, n_actions))
    choice_list.remove(c_action)
    curr_action_ = c_action if rand <= (1 - _EPS) else np.random.choice(choice_list)
    return curr_action_


def find_curr_state_room(c_state, offsets):
    _room = [r for r, offset in enumerate(offsets[1:5]) if c_state < offset][0]
    return _room


def execute_option(
    _EPS, n_episode_length, curr_state, goal_state, curr_option, smdp_q_, index
):

    episode_count = 0
    next_state = 0
    curr_reward = 0
    curr_state_reward = []
    curr_state_timestep = 0
    curr_action = 0
    next_state_q_value = 0
    done = False
    in_goal_state = False

    curr_room_states = curr_option["states"]
    curr_beta = curr_option["beta"]
    curr_policy = curr_option["policy"]
    curr_hallway = curr_option["target"]
    curr_hallway_action = curr_option["t_action"]

    while episode_count < n_episode_length:

        curr_action = curr_policy[curr_state] if curr_state in curr_room_states else 0

        if curr_state != curr_hallway and curr_beta[curr_state] == 1:
            next_state_q_value = max_value(curr_state, smdp_q_)
            break

        elif (
            curr_state == curr_hallway
            and curr_state != goal_state
            and curr_beta[curr_state] == 1
        ):
            curr_action = curr_hallway_action

        elif (
            curr_state == curr_hallway
            and curr_state == goal_state
            and curr_beta[curr_state] == 1
        ):
            next_state_q_value = 0  # 0 max_value(curr_state, smdp_q_)
            # curr_action = 0
            done = True
            in_goal_state = True

        elif (
            curr_state != curr_hallway
            and curr_state == goal_state
            and curr_beta[curr_state] == 1
        ):
            next_state_q_value = 0  # 0 max_value(curr_state, smdp_q_)
            # curr_action = 0
            done = True
            in_goal_state = True

        curr_action = epsilon_greedy_policy(curr_action, _EPS)
        next_state, curr_reward, done, _ = env.step(curr_action)
        curr_state_reward.append(curr_reward)
        curr_state_timestep += 1
        curr_state = next_state
        episode_count += 1

        if in_goal_state == True:
            break

        if episode_count == n_episode_length:
            next_state_q_value = max_value(curr_state, smdp_q_)
            break

    return curr_state, done, next_state_q_value, curr_state_timestep, curr_state_reward


def execute_primitive_action(
    _EPS, n_episode_length, curr_state, goal_state, smdp_q_, offsets
):

    episode_count = 0
    next_state = 0
    curr_reward = 0
    curr_state_reward = []
    curr_state_timestep = 0
    curr_action = 0
    next_state_q_value = 0
    n_room = 0
    c_room = 0
    done = False
    in_goal_state = False

    hallway_states = [25, 56, 77, 103]
    next_room_states = {0: [36, 79], 1: [14, 59], 2: [53, 97], 3: [21, 67]}
    c_room = find_curr_state_room(curr_state, offsets)

    curr_action = max_primitive_action(curr_state, smdp_q_)
    curr_action = epsilon_greedy_policy(curr_action, _EPS)
    curr_state_primitive_action = deepcopy(curr_action)

    while episode_count < n_episode_length:

        if (curr_state in hallway_states) and curr_state == goal_state:
            next_state_q_value = 0  # max_value(curr_state, smdp_q_)
            done = True
            in_goal_state = True

        elif (curr_state not in hallway_states) and curr_state == goal_state:
            next_state_q_value = 0  # max_value(curr_state, smdp_q_)
            done = True
            in_goal_state = True

        next_state, curr_reward, done, _ = env.step(curr_action)
        curr_state_reward.append(curr_reward)
        curr_state_timestep += 1
        curr_state = next_state
        curr_action = max_primitive_action(curr_state, smdp_q_)
        curr_action = epsilon_greedy_policy(curr_action, _EPS)
        episode_count += 1

        if in_goal_state == True:
            break

        n_room = find_curr_state_room(curr_state, offsets)

        if (n_room != c_room) and (curr_state in next_room_states[c_room]):
            next_state_q_value = max_value(curr_state, smdp_q_)
            break

        if episode_count == n_episode_length:
            next_state_q_value = max_value(curr_state, smdp_q_)
            break

    return (
        curr_state,
        done,
        next_state_q_value,
        curr_state_timestep,
        curr_state_primitive_action,
        curr_state_reward,
    )


def learning(
    goal_state,
    alpha,
    gamma,
    n_episode_length,
    n_iterations,
    env_goal,
    offsets,
    states_room_list,
    options_set,
    smdp_q_,
    specific_start_state_bool=False,
    specific_start_state=0,
):

    start_option_list = ["o1", "o2", "primitive_action"]
    specific_start_option_list = ["o7", "o8", "primitive_action"]
    option_list = [start_option_list, specific_start_option_list]
    next_option_dict = {
        14: "o2",
        21: "o1",
        36: "o3",
        53: "o4",
        59: "o5",
        67: "o6",
        79: "o8",
        97: "o7",
    }
    hallway_list = [25, 56, 77, 103]
    clk_list = ["o1", "o3", "o5", "o7"]
    aclk_list = ["o2", "o4", "o6", "o8"]
    _EPS = 0.9

    # First moving in clockwise direction to learn SMDP Q-Learning
    for i in np.arange(n_iterations):

        print("Iteration: ", i)

        if specific_start_state_bool == False:
            opt_list = option_list[0]
        elif specific_start_state_bool == True:
            opt_list = option_list[1]

        for index, opt in enumerate(opt_list):

            if specific_start_state_bool == False:
                state = states_room_list[np.random.randint(0, 25)]
            elif specific_start_state_bool == True:
                state = specific_start_state

            env.reset_state(state, env_goal)
            done = False
            curr_state = deepcopy(state)
            curr_start_state = deepcopy(state)

            if index < 2:
                start_option = options_set[opt]
                curr_option = deepcopy(start_option)

            curr_room = find_curr_state_room(curr_state, offsets)
            curr_state_primitive_action = 0

            while curr_state != goal_state and done == False:
                if index < 2:
                    (
                        curr_state,
                        done,
                        next_state_q_value,
                        curr_state_timestep,
                        curr_state_reward,
                    ) = execute_option(
                        _EPS,
                        n_episode_length,
                        curr_state,
                        goal_state,
                        curr_option,
                        smdp_q_,
                        index,
                    )
                    new_room = find_curr_state_room(curr_state, offsets)

                    if index == 0:
                        curr_q = smdp_q_[curr_start_state][0]
                    elif index == 1:
                        curr_q = smdp_q_[curr_start_state][1]

                    if curr_room != new_room:

                        if (
                            curr_state in next_option_dict.keys()
                            and curr_state != goal_state
                        ):
                            curr_option = options_set[next_option_dict[curr_state]]
                        elif curr_state in hallway_list and curr_room > new_room:
                            curr_option = options_set[aclk_list[new_room]]
                        elif curr_state in hallway_list and curr_room < new_room:
                            curr_option = options_set[clk_list[new_room]]

                        curr_room = new_room

                    elif curr_room == new_room:
                        curr_option = curr_option

                if index == 2:

                    (
                        curr_state,
                        done,
                        next_state_q_value,
                        curr_state_timestep,
                        curr_state_primitive_action,
                        curr_state_reward,
                    ) = execute_primitive_action(
                        _EPS, n_episode_length, curr_state, goal_state, smdp_q_, offsets
                    )

                    curr_q = smdp_q_[curr_start_state][curr_state_primitive_action + 2]

                total_reward_curr_state = calculate_state_option_reward(
                    gamma, curr_state_timestep, curr_state_reward
                )
                curr_q = curr_q + alpha * (
                    (
                        total_reward_curr_state
                        + (math.pow(gamma, curr_state_timestep) * next_state_q_value)
                    )
                    - curr_q
                )

                if index == 0:
                    smdp_q_[curr_start_state][0] = curr_q
                elif index == 1:
                    smdp_q_[curr_start_state][1] = curr_q
                elif index == 2:
                    smdp_q_[curr_start_state][curr_state_primitive_action + 2] = curr_q

                curr_start_state = deepcopy(curr_state)

        _EPS -= 0.1 / n_iterations if _EPS > 0 else 0
        _EPS = round(_EPS, 5)

    return smdp_q_


def get_q_values_list(smdp_list, offsets):
    file_name_list = ["smdp_g1", "smdp_g2", "smdp_g3", "smdp_g4"]

    for index in np.arange(len(smdp_list)):
        q_value_list = []
        smdp_max = get_max_q_values(smdp_list[index])
        (
            q_value_o1,
            q_value_o2,
            q_value_o3,
            q_value_o4,
            q_value_o5,
            q_value_o6,
            q_value_o7,
            q_value_o8,
            q_value_up,
            q_value_right,
            q_value_down,
            q_value_left,
        ) = get_q_values(smdp_list[index], offsets)
        q_value_list.append(smdp_max)
        q_value_list.append(q_value_o1)
        q_value_list.append(q_value_o2)
        q_value_list.append(q_value_o3)
        q_value_list.append(q_value_o4)
        q_value_list.append(q_value_o5)
        q_value_list.append(q_value_o6)
        q_value_list.append(q_value_o7)
        q_value_list.append(q_value_o8)
        q_value_list.append(q_value_up)
        q_value_list.append(q_value_right)
        q_value_list.append(q_value_down)
        q_value_list.append(q_value_left)

        print_heatmap(q_value_list, file_name_list[index])


def get_q_values(smdp_q, offsets):
    q_value_o1 = []
    q_value_o2 = []
    q_value_o3 = []
    q_value_o4 = []
    q_value_o5 = []
    q_value_o6 = []
    q_value_o7 = []
    q_value_o8 = []
    q_value_left = []
    q_value_right = []
    q_value_up = []
    q_value_down = []

    for index, state in enumerate(smdp_q):
        if index >= offsets[0] and index <= offsets[1] - 1:
            q_value_o1.append(state[0])
            q_value_o2.append(state[1])
        else:
            q_value_o1.append(0)
            q_value_o2.append(0)

        if index >= offsets[1] and index <= offsets[2] - 1:
            q_value_o3.append(state[0])
            q_value_o4.append(state[1])
        else:
            q_value_o3.append(0)
            q_value_o4.append(0)

        if index >= offsets[2] and index <= offsets[3] - 1:
            q_value_o5.append(state[0])
            q_value_o6.append(state[1])
        else:
            q_value_o5.append(0)
            q_value_o6.append(0)

        if index >= offsets[3] and index <= offsets[4] - 1:
            q_value_o7.append(state[0])
            q_value_o8.append(state[1])
        else:
            q_value_o7.append(0)
            q_value_o8.append(0)

        q_value_up.append(state[2])
        q_value_right.append(state[3])
        q_value_down.append(state[4])
        q_value_left.append(state[5])

    return (
        q_value_o1,
        q_value_o2,
        q_value_o3,
        q_value_o4,
        q_value_o5,
        q_value_o6,
        q_value_o7,
        q_value_o8,
        q_value_up,
        q_value_right,
        q_value_down,
        q_value_left,
    )


def print_heatmap(q_value_list, file_name):
    print_list = [
        "Max_q_values",
        "option o1",
        "option o2",
        "option o3",
        "option o4",
        "option o5",
        "option o6",
        "option o7",
        "option o8",
        "Primitive_action_UP",
        "Primitive_action_RIGHT",
        "Primitive_action_DOWN",
        "Primitive_action_LEFT",
    ]
    for index in np.arange(len(q_value_list)):
        q_value_heatmap = np.array(q_value_list[index]).reshape(13, 8)
        hmap = sb.heatmap(q_value_heatmap, xticklabels=False, yticklabels=False,)
        text = file_name + "_" + print_list[index]
        plt.title(text)
        plt.show()


smdp_g1 = create_qvalues(n_states, goal_states[0])
smdp_g2 = create_qvalues(n_states, goal_states[1])
smdp_g3 = create_qvalues(n_states, goal_states[0])
smdp_g4 = create_qvalues(n_states, goal_states[1])

offsets = create_offsets(n_rooms, room_sizes)
states_room_list = create_states_rooms(offsets)
policy_list = create_policy_list(states_room_list, policies)
beta_list = create_terminating_condition_list(n_rooms, beta_options, offsets)
options_set = create_options(
    states_room_list, beta_list, policy_list, rooms_options, rooms_states
)


smdp_g1 = learning(
    goal_states[0],
    alpha,
    gamma,
    n_episode_length,
    n_iterations,
    goal1,
    offsets,
    states_room_list[0],
    options_set,
    smdp_g1,
)
smdp_g2 = learning(
    goal_states[1],
    alpha,
    gamma,
    n_episode_length,
    n_iterations,
    goal2,
    offsets,
    states_room_list[0],
    options_set,
    smdp_g2,
)
specific_start_state_bool = True
smdp_g3 = learning(
    goal_states[0],
    alpha,
    gamma,
    n_episode_length,
    n_iterations,
    goal1,
    offsets,
    states_room_list[0],
    options_set,
    smdp_g3,
    specific_start_state_bool,
    specific_start_state,
)
smdp_g4 = learning(
    goal_states[1],
    alpha,
    gamma,
    n_episode_length,
    n_iterations,
    goal2,
    offsets,
    states_room_list[0],
    options_set,
    smdp_g4,
    specific_start_state_bool,
    specific_start_state,
)

smdp_list = [smdp_g1, smdp_g2, smdp_g3, smdp_g4]


get_q_values_list(smdp_list, offsets)

