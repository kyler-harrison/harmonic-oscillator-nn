import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time


def underdamped_position(t_values, amplitude, gamma, Omega, phi):
    """
    Description:
    Calculates the position value of an underdamped harmonic oscillator
    with the inputted t_values and parameters.
    For math, see: https://phys.libretexts.org/Bookshelves/Mathematical_Physics_and_Pedagogy/Complex_Methods_for_the_Sciences_(Chong)/05%3A_Complex_Oscillations/5.03%3A_General_Solution_for_the_Damped_Harmonic_Oscillator

    Inputs:
    t_values: np array or scalar of time values, output has same shape
    amplitude: factor multiplied by entire expression
    gamma: damping coefficient
    Omega: sqrt((angular frequency)**2 - gamma**2). Note that this is 
           capital Omega, not to be confused with lowercase omega 
           (angular frequency). Also note that omega > gamma for 
           underdamped case.
    phi: phase shift

    Output:
    returns np array or scalar of position values, calculated using 
    general solution to underdamped harmonic oscillator (see url in 
    description).

    """
    return amplitude * np.exp(-gamma * t_values) * np.cos(Omega * t_values + phi)


def generate_params(oscillation_type="good"):
    """
    Description:
    Generates random values of Omega, gamma, amplitude, and phi that fit
    constraints imposed by oscillation_type.

    Inputs:
    oscillation_type: either "good" (small gamma) or "quick-damp" (larger
                      gamma)

    Output:
    returns amplitude, gamma, Omega, phi

    """
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

    return amplitude, gamma, Omega, phi


def generate_independent_data(file_path, file_mode, num_samples, t_start=0, t_stop=100, t_step=1, oscillation_type="good"):
    """
    Description:
    Generates a random value of t and random param values, then computes
    the underdamped harmonic oscillator position. Does this num_samples
    times and then writes data to csv.

    Inputs:
    file_path: name of csv to write generated data to
    file_mode: mode to open file_path in
    num_samples: number of random samples to generate (output file will
                 have this many lines +1 for header)
    t_start: beginning value of t range
    t_stop: ending value of t range
    oscillation_type: either "good" (small gamma) or "quick-damp" (larger
                      gamma)

    Output:
    returns None, writes data to file
    """
    generated_samples = []
    t_choices = np.arange(t_start, t_stop + t_step, t_step)

    for i in range(num_samples):
        t = np.random.choice(t_choices)
        amplitude, gamma, Omega, phi = generate_params(oscillation_type)
        position = underdamped_position(t, amplitude, gamma, Omega, phi)
        generated_samples.append([t, Omega, gamma, phi, amplitude, position])

    with open(file_path, file_mode) as fi:
        if (file_mode == "w"):
            fi.write("t,Omega,gamma,phi,amplitude,y\n")

        for sample in generated_samples:
            fi.write(f"{','.join([str(val) for val in sample])}\n")


def generate_range_data(file_path, file_mode, t_start=0, t_stop=100, t_step=1, oscillation_type="good"):
    """
    Description: generates a set of underdamped harmonic oscillator data
                 using a fixed range of t and random constants.

    Inputs:
    file_path: where to write data to 
    file_mode: mode to open data file in
    t_start: starting value of t (defaults to 0)
    t_stop: stopping value of t (defaults to 100)
    t_step: number of values from t_start to t_stop (defaults to 1000)
    oscillation_type: "good" or "quick-damp"

    Output:
    returns None, writes data to file_path
    """
    # hold range of t values constant across all generated data
    # this creates 1000 samples
    t_values = np.arange(t_start, t_stop + t_step, t_step)

    # randomly generate parameters
    amplitude, gamma, Omega, phi = generate_params(oscillation_type)

    # analytical solution (y values)
    positions = underdamped_position(t_values, amplitude, gamma, Omega, phi)

    # write params and y vs. t to csv
    with open(file_path, file_mode) as fi:
        if file_mode == "w":
            fi.write("t,Omega,gamma,phi,amplitude,y\n")

        for i in range(len(t_values)):
            fi.write(f"{t_values[i]},{Omega},{gamma},{phi},{amplitude},{positions[i]}\n")


def main():
    data_dir = "data"
    os.system(f"mkdir -p {data_dir}")
    create_range = True

    if create_range:
        # create data in range t_0 to t_f with constant params
        # creates multiple ranges with different params

        num_datasets = 1
        dataset_type = "underdamped_range_validation"
        t_start = 0
        t_stop = 50
        t_step = 0.1
        oscillation_type = "good"
        file_path = f"{data_dir}/{dataset_type}_{oscillation_type}_t0={t_start}_tf={t_stop}_ts={t_step}.csv"
        file_mode = "w"

        # create num_datasets of "good" damping
        for i in range(num_datasets):
            generate_range_data(file_path, file_mode, t_start=t_start, t_stop=t_stop, t_step=t_step, oscillation_type="good")
            file_mode = "a"

        """
        # create num_datasets of "quick-damp" damping
        for i in range(num_datasets):
            generate_range_data(file_path, file_mode, oscillation_type="quick-damp")
        """

    else:
        # create random t and random params, compute position
        # create num_samples of these (t constrained to t_0 to t_f, just 
        # not generating entire range with same constant params)
        start = time.time()

        num_samples = int(1e4)
        dataset_type = "underdamped_independent_test"
        t_start = 0
        t_stop = 50
        t_step = 0.1
        oscillation_type = "good"

        file_path = f"{data_dir}/{dataset_type}_{oscillation_type}_t0={t_start}_tf={t_stop}_ts={t_step}.csv"
        file_mode = "w"
        generate_independent_data(file_path, file_mode, num_samples, t_start=t_start, t_stop=t_stop, t_step=t_step, oscillation_type=oscillation_type)

        print(f"time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()

