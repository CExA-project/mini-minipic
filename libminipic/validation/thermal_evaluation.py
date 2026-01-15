"""Validation script for the `default` setup."""

import os

import numpy as np

from libminipic import ci as minipic_ci
from libminipic import diag as minipic_diag
from libminipic.exceptions import MissingFileMiniPICError
from libminipic.validate import THRESHOLD


def validate(evaluate=True, threshold=THRESHOLD):

    # ______________________________________________________________________
    # Check output files are created

    # list of output files
    output_file_list = []

    # Add *.vtk files
    for field in [
        "Ex",
        "Ey",
        "Ez",
        "Bx",
        "By",
        "Bz",
        "diag_x_y_z_d_s00",
        "diag_x_y_z_d_s01",
        "diag_px_py_pz_d_s00",
        "diag_px_py_pz_d_s01",
    ]:
        for it in range(0, 200, 200):
            file = "{}_{:03d}.vtk".format(field, it)
            output_file_list.append(file)

    # Add *.bin files
    for field in ["cloud_s00", "cloud_s01", "diag_w_gamma_s00", "diag_w_gamma_s01"]:
        for it in range(200, 200, 200):
            file = "{}_{:03d}.bin".format(field, it)
            output_file_list.append(file)

    # Add scalars
    output_file_list.append("fields.txt")
    output_file_list.append("species_00.txt")
    output_file_list.append("species_01.txt")

    # Check that all output files exist
    for file in output_file_list:
        if not (os.path.exists("diags/" + file)):
            raise MissingFileMiniPICError(f"File {file} not generated")

    # ______________________________________________________________________
    # Check scalars

    print(" > Checking scalars")

    # Check initial scalar for species

    reference_data = [
        [0, 14155776.0, 1.499888015286549e-07],
        [0, 14155776.0, 1.50003294849057e-07],
    ]

    for ispecies in range(2):

        with open("diags/species_{:02d}.txt".format(ispecies), "r") as f:
            lines = f.readlines()

            last_line = lines[1].split(" ")

            iteration = int(last_line[0])
            particles = float(last_line[1])
            energy = float(last_line[2])

        print(
            "    - Initial scalar for species {}: {}, {}, {}".format(
                ispecies, iteration, particles, energy
            )
        )

        if evaluate:

            minipic_ci.evaluate(
                iteration,
                reference_data[ispecies][0],
                reference_data[ispecies][0],
                "==",
                "First iteration in species_{}.txt is not correct".format(ispecies),
            )

            minipic_ci.evaluate(
                particles,
                reference_data[ispecies][1],
                reference_data[ispecies][1],
                "==",
                "Number of particles in species_{:02d}.txt is not correct".format(
                    ispecies
                ),
            )

            minipic_ci.evaluate(
                energy,
                reference_data[ispecies][2],
                threshold,
                "relative",
                "Kinetic energy in species_{:02d}.txt is not correct".format(ispecies),
            )

    # Check final scalar for species

    reference_data = [
        [200, 14155776.0, 1.49989149925599e-07],
        [200, 14155776.0, 1.500033300017442e-07],
    ]

    for ispecies in range(2):

        with open("diags/species_{:02d}.txt".format(ispecies), "r") as f:
            lines = f.readlines()

            last_line = lines[-1].split(" ")

            iteration = int(last_line[0])
            particles = float(last_line[1])
            energy = float(last_line[2])

        print(
            "    - Final scalar for species {}: {}, {}, {}".format(
                ispecies, iteration, particles, energy
            )
        )

        if evaluate:

            minipic_ci.evaluate(
                iteration,
                reference_data[ispecies][0],
                reference_data[ispecies][0],
                "==",
                "Last iteration in species_{}.txt is not correct".format(ispecies),
            )

            minipic_ci.evaluate(
                particles,
                reference_data[ispecies][1],
                reference_data[ispecies][1],
                "==",
                "Number of particles in species_{:02d}.txt is not correct".format(
                    ispecies
                ),
            )

            minipic_ci.evaluate(
                energy,
                reference_data[ispecies][2],
                threshold,
                "relative",
                "Kinetic energy in species_{:02d}.txt is not correct".format(ispecies),
            )

    # Check final scalar values for fields

    reference_data = [
        200,
        2.342884983394541e-17, 2.408514291695298e-17, 2.315415706915853e-17, 3.16080235073564e-18, 2.657409335570545e-18, 2.248892086147286e-18
    ]

    with open("diags/fields.txt", "r") as f:
        lines = f.readlines()

        last_line = lines[-1].split(" ")

        iteration = int(last_line[0])
        Ex = float(last_line[1])
        Ey = float(last_line[2])
        Ez = float(last_line[3])
        Bx = float(last_line[4])
        By = float(last_line[5])
        Bz = float(last_line[6])

    print(
        "    - Field values at final iteration {}: \n {}, {}, {}, {}, {}, {}".format(
            iteration, Ex, Ey, Ez, Bx, By, Bz
        )
    )

    if evaluate:

        minipic_ci.evaluate(
            iteration,
            reference_data[0],
            reference_data[0],
            "==",
            "Last iteration in fields.txt is not correct",
        )

        # We check that the fields do not explode (numerical instability)
        minipic_ci.evaluate(
            Ex,
            reference_data[1],
            threshold,
            "relative",
            "Ex value at it {} in fields.txt is not correct".format(iteration),
        )
        minipic_ci.evaluate(
            Ey,
            reference_data[2],
            threshold,
            "relative",
            "Ey value at it {} in fields.txt is not correct".format(iteration),
        )
        minipic_ci.evaluate(
            Ez,
            reference_data[3],
            threshold,
            "relative",
            "Ez value at it {} in fields.txt is not correct".format(iteration),
        )
        minipic_ci.evaluate(
            Bx,
            reference_data[4],
            threshold,
            "relative",
            "Bx value at it {} in fields.txt is not correct".format(iteration),
        )
        minipic_ci.evaluate(
            By,
            reference_data[5],
            threshold,
            "relative",
            "By value at it {} in fields.txt is not correct".format(iteration),
        )
        minipic_ci.evaluate(
            Bz,
            reference_data[6],
            threshold,
            "relative",
            "Bz value at it {} in fields.txt is not correct".format(iteration),
        )

    # ______________________________________________________________________
    # Check gamma spectrums

    reference_sum_data = [
        [
            1.0150531773248289e-05
        ],
        [
            1.0001577013349268e-05
        ],
    ]

    print(" > Checking gamma spectrums")

    for ispecies in range(2):

        new_data = []

        for i, it in enumerate(range(0, 200, 200)):

            file = "diag_w_gamma_s{:02d}_{:03d}.bin".format(ispecies, it)

            x_axis_name, x_min, x_max, x_data, data_name, data = (
                minipic_diag.read_1d_diag("diags/" + file)
            )

            new_data.append(np.sum(np.abs(data * x_data)))

        print("    - For species {}: {}".format(ispecies, new_data))

        if evaluate:
            for i, it in enumerate(range(0, 200, 200)):
                minipic_ci.evaluate(
                    new_data[i],
                    reference_sum_data[ispecies][i],
                    1e-13,
                    "relative",
                    "Gamma spectrum for species {} at iteration {} not similar".format(
                        ispecies, it
                    ),
                )

    # ______________________________________________________________________
    # Check initial cloud file (particle initialization)

    reference_data = [
        [
           7077889.707965645, 7077888.015571134, 7077886.520995862, 1135057.9472597158, 1135056.6569268475, 1135050.3021177554

        ],
        [
           7077889.707965645, 7077888.015571134, 7077886.520995862, 26358.85248197128, 26359.12372698904, 26359.11314302442
        ],
    ]

    print(" > Checking initial cloud file")

    for ispecies in range(2):

        file = "cloud_s{:02}_000.bin".format(ispecies)

        particle_number, id, w, x, y, z, px, py, pz = minipic_diag.read_particle_cloud(
            "diags/" + file
        )

        x_sum = np.sum(np.abs(x))
        y_sum = np.sum(np.abs(y))
        z_sum = np.sum(np.abs(z))

        px_sum = np.sum(np.abs(px))
        py_sum = np.sum(np.abs(py))
        pz_sum = np.sum(np.abs(pz))

        print(
            "   - For Species {}: {}, {}, {}, {}, {}, {}".format(
                ispecies, x_sum, y_sum, z_sum, px_sum, py_sum, pz_sum
            )
        )

        if evaluate:
            minipic_ci.evaluate(
                x_sum,
                reference_data[ispecies][0],
                threshold,
                "relative",
                "Sum over x positions not similar",
            )
            minipic_ci.evaluate(
                y_sum,
                reference_data[ispecies][1],
                threshold,
                "relative",
                "Sum over y positions not similar",
            )
            minipic_ci.evaluate(
                z_sum,
                reference_data[ispecies][2],
                threshold,
                "relative",
                "Sum over z positions not similar",
            )

            minipic_ci.evaluate(
                px_sum,
                reference_data[ispecies][3],
                threshold,
                "relative",
                "Sum over px not similar",
            )
            minipic_ci.evaluate(
                py_sum,
                reference_data[ispecies][4],
                threshold,
                "relative",
                "Sum over py not similar",
            )
            minipic_ci.evaluate(
                pz_sum,
                reference_data[ispecies][5],
                threshold,
                "relative",
                "Sum over pz not similar",
            )

    # ______________________________________________________________________
    # Check final cloud file (particle initialization)

    reference_data = [
        [
            7077877.48158804, 7078261.1447573025, 7078347.829847816, 1135065.017936733, 1135054.7563931916, 1135052.031481696

        ],
        [
            7075641.898230777, 7078221.58157913, 7077548.476066667, 26358.887515847564, 26359.125779929567, 26359.080943359495
        ],
    ]

    print(" > Checking final cloud file")

    for ispecies in range(2):

        file = "cloud_s{:02}_200.bin".format(ispecies)

        particle_number, id, w, x, y, z, px, py, pz = minipic_diag.read_particle_cloud(
            "diags/" + file
        )

        x_sum = np.sum(np.abs(x))
        y_sum = np.sum(np.abs(y))
        z_sum = np.sum(np.abs(z))

        px_sum = np.sum(np.abs(px))
        py_sum = np.sum(np.abs(py))
        pz_sum = np.sum(np.abs(pz))

        print(
            "    - For Species {}: {}, {}, {}, {}, {}, {}".format(
                ispecies, x_sum, y_sum, z_sum, px_sum, py_sum, pz_sum
            )
        )

        if evaluate:
            minipic_ci.evaluate(
                x_sum,
                reference_data[ispecies][0],
                threshold,
                "relative",
                "Sum over x positions not similar",
            )
            minipic_ci.evaluate(
                y_sum,
                reference_data[ispecies][1],
                threshold,
                "relative",
                "Sum over y positions not similar",
            )
            minipic_ci.evaluate(
                z_sum,
                reference_data[ispecies][2],
                threshold,
                "relative",
                "Sum over z positions not similar",
            )

            minipic_ci.evaluate(
                px_sum,
                reference_data[ispecies][3],
                threshold,
                "relative",
                "Sum over px not similar",
            )
            minipic_ci.evaluate(
                py_sum,
                reference_data[ispecies][4],
                threshold,
                "relative",
                "Sum over py not similar",
            )
            minipic_ci.evaluate(
                pz_sum,
                reference_data[ispecies][5],
                threshold,
                "relative",
                "Sum over pz not similar",
            )
