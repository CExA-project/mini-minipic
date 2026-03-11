"""Validation script for the `Ecst` setup."""

import os

import numpy as np

from libminipic import ci as minipic_ci
from libminipic import diag as minipic_diag
from libminipic.exceptions import IncorrectValueMiniPICError, MissingFileMiniPICError
from libminipic.validate import THRESHOLD


def validate(evaluate=True, threshold=THRESHOLD):

    # Get the name of the current file (without extension) to use as a label for the validation
    validation_label = os.path.basename(__file__).split(".")[0]

    number_of_iterations = 50000

    # ______________________________________________________________________
    # Check output files are created

    # list of output files
    output_file_list = []

    # Create output file list
    for field in ["cloud_s00"]:
        for it in range(0, number_of_iterations, 1000):

            file = "{}_{:05d}.bin".format(field, it)
            output_file_list.append(file)

    # Add scalars
    output_file_list.append("fields.txt")
    output_file_list.append("species_00.txt")

    # Check that all output files exist
    for file in output_file_list:
        if not (os.path.exists("diags/" + file)):
            raise MissingFileMiniPICError(f"File {file} not generated")

    # ______________________________________________________________________
    # Reference database

    references = {
        "species_final": [50000, 1.0, 0.07602920573221358],
        "cloud": [
            24.411730435676887,
            23.88234608876007,
            23.05876871008283,
            994.5565640860931,
            198.9113128173085,
            1591.290502538468,
        ],
    }

    # ______________________________________________________________________
    # Check final scalar for species

    reference_data = references["species_final"]

    with open("diags/species_00.txt", "r") as f:
        lines = f.readlines()

        last_line = lines[-1].split(" ")

        iteration = int(last_line[0])
        particles = float(last_line[1])
        energy = float(last_line[2])

    if evaluate:

        if reference_data[0] != iteration:
            raise IncorrectValueMiniPICError(
                f"Last iteration in species_00.txt is not correct: {iteration} instead of {reference_data[0]}"
            )

        minipic_ci.evaluate(
            particles,
            reference_data[1],
            reference_data[1],
            "==",
            f"Number of particles in species_00.txt is not correct: {particles} instead of {reference_data[1]}",
        )

        minipic_ci.evaluate(
            energy,
            reference_data[2],
            threshold,
            "relative",
            f"Kinetic energy in species_00.txt is not correct: {energy}, {reference_data[2]}",
        )

    else:

        references["species_final"] = [iteration, particles, energy]

    # ______________________________________________________________________
    # Check cloud files

    reference_cloud = references["cloud"]

    nb_files = int(number_of_iterations / 1000)

    x_array = np.zeros(nb_files)
    y_array = np.zeros(nb_files)
    z_array = np.zeros(nb_files)

    px_array = np.zeros(nb_files)
    py_array = np.zeros(nb_files)
    pz_array = np.zeros(nb_files)

    for i, it in enumerate(range(0, number_of_iterations, 1000)):

        file = "cloud_s00_{:05d}.bin".format(it)

        particle_number, id, w, x, y, z, px, py, pz = minipic_diag.read_particle_cloud(
            "diags/" + file
        )

        x_array[i] = x[0]
        y_array[i] = y[0]
        z_array[i] = z[0]

        px_array[i] = px[0]
        py_array[i] = py[0]
        pz_array[i] = pz[0]

    x_sum = np.sum(np.abs(x_array))
    y_sum = np.sum(np.abs(y_array))
    z_sum = np.sum(np.abs(z_array))

    px_sum = np.sum(np.abs(px_array))
    py_sum = np.sum(np.abs(py_array))
    pz_sum = np.sum(np.abs(pz_array))

    if evaluate:

        minipic_ci.evaluate(
            x_sum, reference_cloud[0], threshold, "relative", "Sum over x positions not similar"
        )
        minipic_ci.evaluate(
            y_sum, reference_cloud[1], threshold, "relative", "Sum over y positions not similar"
        )
        minipic_ci.evaluate(
            z_sum, reference_cloud[2], threshold, "relative", "Sum over z positions not similar"
        )

        minipic_ci.evaluate(
            px_sum, reference_cloud[3], threshold, "relative", "Sum over px not similar"
        )
        minipic_ci.evaluate(
            py_sum, reference_cloud[4], threshold, "relative", "Sum over py not similar"
        )
        minipic_ci.evaluate(
            pz_sum, reference_cloud[5], threshold, "relative", "Sum over pz not similar"
        )

    else:

        references["cloud"] = [x_sum, y_sum, z_sum, px_sum, py_sum, pz_sum]

    # ______________________________________________________________________
    # Write updated references to JSON file if not evaluating

    if not evaluate:
        import json

        json_file_name = validation_label

        with open(json_file_name, "w") as f:
            json.dump(references, f, indent=4)
        print(f" > New references written to {json_file_name}")

        print(
            " > New references: \n {}".format(json.dumps(references, indent=4))
        )
