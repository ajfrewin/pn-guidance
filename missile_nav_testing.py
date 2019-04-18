"""
File Name: missile_nav_testing.py
Author: Adam Frewin
Date Last Modified: April 17th 2019

This code runs a simulation of a missile strike event, using proportional navigation law for missile guidance.

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as ani
from math import sin, cos, atan, asin, sqrt

TARGET_MANEUVER = {0: [0, 0, 0], 1:[0, .05, 0], 2:[0, -0.05, -0.05], 3:[0, 0, -0.09]}

def rK4_step(state, derivs, dt, params=None, time_invariant=True, t=0):
    # One step of time-invariant RK4 integration
    if params is not None:
        k1 = dt * derivs(state, params)
        k2 = dt * derivs(state + k1 / 2, params)
        k3 = dt * derivs(state + k2 / 2, params)
        k4 = dt * derivs(state + k3, params)

        return state + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    else:
        k1 = dt * derivs(state)
        k2 = dt * derivs(state + k1 / 2)
        k3 = dt * derivs(state + k2 / 2)
        k4 = dt * derivs(state + k3)

        return state + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def acquisition(target_pos):
    """
    Sets initial heading and climb angle to LOS angles, better ensuring a collision
    :param target_pos: initial detected position of target
    :return: recommended initial climb angle and heading angle
    """
    Rx = target_pos[0]
    Ry = target_pos[1]
    Rz = target_pos[2]

    Rxy = np.linalg.norm([Rx, Ry])
    R = np.linalg.norm(target_pos)

    psi_rec = asin(Ry/Rxy)
    gma_rec = -asin(Rz/R)
    return [psi_rec, gma_rec]


def target_accel(target_state, target_maneuver):
    """
    Computes state derivative of the target, currently no acceleration
    :param target_state: the current state of the target, Px, Py, Pz, V, psi, gamma
    :return: the state derivative
    """

    # Target velocity and heading
    VT = target_state[3]
    psi_t = target_state[4]
    gma_t = target_state[5]

    Rie_t = np.array([[cos(gma_t), 0, -sin(gma_t)],
                      [0, 1, 0],
                      [sin(gma_t), 0, cos(gma_t)]]).dot(
        np.array([[cos(psi_t), sin(psi_t), 0],
                  [-sin(psi_t), cos(psi_t), 0],
                  [0, 0, 1]]))

    a_t = TARGET_MANEUVER[target_maneuver]
    VT_inert = Rie_t.T.dot(np.array([VT, 0, 0]).T)

    return np.array([VT_inert[0], VT_inert[1], VT_inert[2], a_t[0], a_t[1], a_t[2]])


def missile_accel(missile_state, commanded_accel):
    """
    Computes state derivative of the missile given a commanded acceleration
    :param missile_state: the current state of the missile, Px, Py, Pz, V, psi, gamma
    :param commanded_accel: inertial coordinates of commanded acceleration
    :return: actual derivative of the missile state
    """
    Vm = missile_state[3]
    psi_m = missile_state[4]
    gma_m = missile_state[5]

    Rie_m = np.array([[cos(gma_m), 0, -sin(gma_m)],
                      [0, 1, 0],
                      [sin(gma_m), 0, cos(gma_m)]]).dot(
            np.array([[cos(psi_m), sin(psi_m), 0],
                      [-sin(psi_m), cos(psi_m), 0],
                      [0, 0, 1]]))

    Vm_inert = Rie_m.T.dot(np.array([Vm, 0, 0]).T)

    ac_body_fixed = Rie_m.dot(commanded_accel.T)

    ax = ac_body_fixed[0]
    ay = ac_body_fixed[1]
    az = ac_body_fixed[2]

    return np.array([Vm_inert[0], Vm_inert[1], Vm_inert[2],
                     ax, ay/(Vm*cos(gma_m)), -az/Vm])


def get_seeker_state(target_state, missile_state):
    """
    Computes the "seeker state," relative position and velocity in inertial coordinates
    :param target_state: position, speed, and orientation of target
    :param missile_state: position, speed, and orientation of missile
    :return: the seeker state, 3 coordinates of inertial position, 3 components of inertial velocity
    """
    # define variables for readability
    # T = target
    # M = missile

    # Target position
    RTx = target_state[0]
    RTy = target_state[1]
    RTz = target_state[2]

    # Target velocity and heading
    VT = target_state[3]
    psi_t = target_state[4]
    gma_t = target_state[5]

    # Missile Position
    RMx = missile_state[0]
    RMy = missile_state[1]
    RMz = missile_state[2]

    # Missile velocity and heading
    VM = missile_state[3]
    psi_m = missile_state[4]
    gma_m = missile_state[5]

    # Rotation matrices from inertial to body fixed coordinates for
    # both the target and the missile
    Rie_t = np.array([[cos(gma_t), 0, -sin(gma_t)],
                      [0, 1, 0],
                      [sin(gma_t), 0, cos(gma_t)]]).dot(
        np.array([[cos(psi_t), sin(psi_t), 0],
                  [-sin(psi_t), cos(psi_t), 0],
                  [0, 0, 1]]))

    Rie_m = np.array([[cos(gma_m), 0, -sin(gma_m)],
                      [0, 1, 0],
                      [sin(gma_m), 0, cos(gma_m)]]).dot(
        np.array([[cos(psi_m), sin(psi_m), 0],
                  [-sin(psi_m), cos(psi_m), 0],
                  [0, 0, 1]]))


    # get relative velocity in inertial coordinates
    VT_inert = Rie_t.T.dot(np.array([VT, 0, 0]).T)
    VM_inert = Rie_m.T.dot(np.array([VM, 0, 0]).T)

    return np.array([RTx - RMx, RTy - RMy, RTz - RMz,
                     VT_inert[0] - VM_inert[0], VT_inert[1] - VM_inert[1], VT_inert[2] - VM_inert[2]])


def pn_guidance(seeker_state):
    """
    Provides commanded accelerations using proportional navigation law
    :param seeker_state: current seeker state
    :return: Commanded accelerations in inertial coordinates
    """

    Rx = seeker_state[0]
    Ry = seeker_state[1]
    Rz = seeker_state[2]
    Vx = seeker_state[3]
    Vy = seeker_state[4]
    Vz = seeker_state[5]

    R = np.linalg.norm([Rx, Ry, Rz])
    Rxy = np.linalg.norm([Rx,Ry])

    # alpha = angle from x axis to xy projection of R
    # beta = angle from R down to xy plane
    alpha = asin(Ry / Rxy)
    alpha_dot = (Rxy * Vy - Ry * (Rx * Vx + Ry * Vy) / Rxy) / (Rxy ** 2)

    beta = asin(Rz / R)

    # closing velocity, negative of the rate of change of LOS
    Vc = -(Rx * Vx + Ry * Vy + Rz * Vz) / R
    beta_dot = (R * Vz + Vc * Rz) / (R ** 2)

    # Proportional navigation law for horizontal and vertical accelerations
    ach_mag = 3 * Vc * alpha_dot
    acv_mag = 3 * Vc * beta_dot

    # resolve accelerations into xyz components
    ach = np.array([-ach_mag * sin(alpha), ach_mag * cos(alpha), 0])
    acv = np.array([acv_mag * sin(beta) * sin(alpha), -acv_mag * sin(beta) * cos(alpha), acv_mag * cos(beta)])

    return ach + acv

def missile_strike_scenario():

    ## Script

    # Target will start in one position
    RT_0 = np.array([2000, 1000, 1500])

    VT_0 = 210
    psi_t_0 = 0.2
    gma_t_0 = 0
    target_state_0 = np.array([*RT_0, VT_0, psi_t_0, gma_t_0])

    # Target Maneuver, 0 = line, 1 = circle, 2 = Helix, 3 = loop
    TARGET_MANEUVER_NUM = 0

    # missile starts at the same spot with the same velocity
    RM_0 = np.array([0, 0, 0])
    VM_0 = 400
    engage_params = acquisition(RT_0)
    psi_m_0 = engage_params[0]
    gma_m_0 = engage_params[1]

    missile_state_0 = np.array([*RM_0, VM_0, psi_m_0, gma_m_0])

    seeker_state_0 = get_seeker_state(target_state_0, missile_state_0)

    # set up loop to stop when contact is made, or negative Vc
    t = 0
    dt = 0.005
    missile_state = missile_state_0
    target_state = target_state_0
    t_sim = [t]
    missile_sim = np.array(missile_state)  # ACTUAL MISSILE STATE
    target_sim = np.array(target_state)  # ACTUAL TARGET STATE
    distance = np.linalg.norm(missile_state[:3]-target_state[:3])
    distace_sim = [distance]
    seeker_state = seeker_state_0

    eps = 1. # impact distance parameter

    Vc_track = -(seeker_state[0] * seeker_state[3] + seeker_state[1] * seeker_state[4] + seeker_state[2] * seeker_state[5]) \
               / np.linalg.norm(seeker_state[:3])

    while (distance > eps and Vc_track > 0.):

        commanded_accel = pn_guidance(seeker_state)

        missile_state = rK4_step(missile_state, missile_accel, dt, params=commanded_accel)
        target_state = rK4_step(target_state, target_accel, dt, params=TARGET_MANEUVER_NUM)
        seeker_state = get_seeker_state(target_state, missile_state)

        distance = np.linalg.norm(missile_state[:3] - target_state[:3])
        Vc_track = -(seeker_state[0] * seeker_state[3] + seeker_state[1] * seeker_state[4] + seeker_state[2] * seeker_state[
            5]) / np.linalg.norm(seeker_state[:3])

        t = t+dt
        t_sim.append(t)
        missile_sim = np.vstack((missile_sim, missile_state))
        target_sim = np.vstack((target_sim, target_state))
        distace_sim.append(distance)

    # Check if hit was successful
    if(distance<=eps):
        hit = True
    else:
        hit = False

    return target_sim, missile_sim, hit


# Animation update
def update_lines(num, target_data, missile_data, line_target, line_missile):
    # NOTE: there is no .set_data() for 3 dim data...
    line_target.set_data([target_data[:num, 0], target_data[:num,1]])
    line_target.set_3d_properties(target_data[:num, 2])
    line_missile.set_data([missile_data[:num, 0], missile_data[:num,1]])
    line_missile.set_3d_properties(missile_data[:num, 2])
    return line_target, line_missile


target_sim, missile_sim, hit_success = missile_strike_scenario()

plot_scale_factor = 30 # To speed up animation, we dont need to animate every single time-step
target_path_plot = np.vstack((target_sim[::plot_scale_factor], target_sim[-1]))
missile_path_plot = np.vstack((missile_sim[::plot_scale_factor], missile_sim[-1]))

ANIMATE = True
if ANIMATE:
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    line_target, = ax.plot(target_sim[0:1,0],target_sim[0:1,1],target_sim[0:1,2], linewidth=2)
    line_missile, = ax.plot(missile_sim[0:1,0],missile_sim[0:1,1],missile_sim[0:1,2], linewidth=2)


    N = len(target_path_plot)

    Writer = ani.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

    anim = ani.FuncAnimation(fig, update_lines, N, fargs=(target_path_plot, missile_path_plot, line_target, line_missile),
                             interval=10, blit=False, repeat_delay=2000)


    ax.set_xlim3d([min(np.min(target_sim[:, 0]), np.min(missile_sim[:, 0])),
                   max(np.max(target_sim[:, 0]), np.max(missile_sim[:, 0]))])

    ax.set_ylim3d([min(np.min(target_sim[:, 1]), np.min(missile_sim[:, 1])),
                   max(np.max(target_sim[:, 1]), np.max(missile_sim[:, 1]))])

    ax.set_zlim3d([min(np.min(target_sim[:, 2]), np.min(missile_sim[:, 2])),
                   max(np.max(target_sim[:, 2]), np.max(missile_sim[:, 2]))])



# anim.save('MissileStrikeSim.mp4', writer=writer)


plt.show()