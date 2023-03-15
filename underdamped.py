import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def underdamped_sim(t_values, amplitude, gamma, Omega, phi):
    """
    Description:
    for math, see: https://phys.libretexts.org/Bookshelves/Mathematical_Physics_and_Pedagogy/Complex_Methods_for_the_Sciences_(Chong)/05%3A_Complex_Oscillations/5.03%3A_General_Solution_for_the_Damped_Harmonic_Oscillator

    constraints

    Inputs:
    t_values: np array of time values, output has same shape
    amplitude: factor multiplied by entire expression
    gamma: damping coefficient
    Omega: sqrt((angular frequency)**2 - gamma**2). Note that this is 
           capital Omega, not to be confused with lowercase omega 
           (angular frequency). Also note that omega > gamma for 
           underdamped case.
    phi: phase shift

    Output:
    np array of position values, calculated using general solution to 
    underdamped harmonic oscillator (see url in description).

    """

    return amplitude * np.exp(-gamma * t_values) * np.cos(Omega * t_values + phi)


def generate_data(file_path, file_mode, t_start=0, t_stop=100, t_steps=1000, oscillation_type="good"):
    """
    Description: generates a set of underdamped harmonic oscillator data
                 using a fixed range of t and random constants.

    Inputs:
    file_path: where to write data to 
    file_mode: mode to open data file in
    t_start: starting value of t (defaults to 0)
    t_stop: stopping value of t (defaults to 100)
    t_steps: number of values from t_start to t_stop (defaults to 1000)
    oscillation_type: "good" or "quick-damp"

    Output:
    returns None, writes data to file_path
    """

    # hold range of t values constant across all generated data
    # this creates 1000 samples
    t_values = np.linspace(t_start, t_stop, t_steps)

    if (oscillation_type == "good"):
        # ensure that omega_0 and gamma are 1 order of magnitude different
        # NOTE omega_0 somewhat arbitrarily constrained to range 0.2-1 
        omega_0 = (1 - 0.2) * np.random.random() + 0.2
        gamma = (0.1 - 0.01) * np.random.random() + 0.01
    elif (oscillation_type == "quick-damp"):
        # omega_0 and gamma will be within (closeness * 10)% of each other
        closeness = 0.9
        upper_limit = np.random.random()
        lower_limit = upper_limit - closeness * upper_limit  
        omega_0 = (upper_limit - lower_limit) * np.random.random() + lower_limit
        gamma = (upper_limit - lower_limit) * np.random.random() + lower_limit
        
        # underdamped constraint is omega_0 > gamma
        while (gamma > omega_0):
            gamma = (upper_limit - lower_limit) * np.random.random() + lower_limit

    amplitude = np.random.random()  # range of 0-1
    phi = np.random.random()  # range of 0-1
    Omega = np.sqrt(omega_0**2 - gamma**2)

    # analytical solution (y values)
    positions = underdamped_sim(t_values, amplitude, gamma, Omega, phi)

    # write params and y vs. t to csv
    with open(file_path, file_mode) as fi:
        if file_mode == "w":
            fi.write("t,Omega,gamma,phi,amplitude,y\n")

        for i in range(len(t_values)):
            fi.write(f"{t_values[i]},{Omega},{gamma},{phi},{amplitude},{positions[i]}\n")


def main():
    num_datasets = 100
    file_path = "underdamped_training_data.csv"
    file_mode = "w"

    # create num_datasets of "good" damping
    for i in range(num_datasets):
        generate_data(file_path, file_mode, oscillation_type="good")
        file_mode = "a"

    # create num_datasets of "quick-damp" damping
    for i in range(num_datasets):
        generate_data(file_path, file_mode, oscillation_type="quick-damp")


if __name__ == "__main__":
    main()

