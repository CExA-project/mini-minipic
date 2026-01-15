"""Validation script for the `antenna` setup."""

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

    # Create output file list
    for field in ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]:
        for it in range(0, 201, 200):

            file = "{}_{:03d}.bin".format(field, it)
            output_file_list.append(file)

    # Add scalars
    output_file_list.append("fields.txt")

    # Check that all output files exist
    for file in output_file_list:
        if not (os.path.exists("diags/" + file)):
            raise MissingFileMiniPICError(f"File {file} not generated")

    # ______________________________________________________________________
    # Check scalars

    print(" > Check scalars")

    # Check final scalar values for fields

    reference_data = [
        200,
        1.178332626270596e-08,
        1.121385881385788e-10,
        1.372260141937414e-06,
        1.217329509377536e-08,
        1.397861526161067e-06,
        5.139203602251246e-39,
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

    if evaluate:

        minipic_ci.evaluate(
            iteration,
            reference_data[0],
            threshold,
            "relative",
            "Last iteration in fields.txt is not correct",
        )

        minipic_ci.evaluate(
            Ex,
            reference_data[1],
            threshold,
            "relative",
            "Ex value in fields.txt is not correct",
        )
        minipic_ci.evaluate(
            Ey,
            reference_data[2],
            threshold,
            "relative",
            "Ey value in fields.txt is not correct",
        )
        minipic_ci.evaluate(
            Ez,
            reference_data[3],
            threshold,
            "relative",
            "Ez value in fields.txt is not correct",
        )
        minipic_ci.evaluate(
            Bx,
            reference_data[4],
            threshold,
            "relative",
            "Bx value in fields.txt is not correct",
        )
        minipic_ci.evaluate(
            By,
            reference_data[5],
            threshold,
            "relative",
            "By value in fields.txt is not correct",
        )
        # minipic_ci.evaluate(Bz, reference_data[6], 1e-2, 'relative', 'Bz value in fields.txt is not correct') ===> too small to be relevant

    # ______________________________________________________________________
    # Check field

    print(" > Check fields")

    reference_data = {
        "Ex": [
            0.0,
            78.55592092043187,
        ],
        "Ey": [
            0.0,
            9.5656799728717190,
        ],
        "Ez": [
            0.0,
            748.3567468708843,
        ],
        "Bx": [
            0.0,
            80.94553055070384,
        ],
        "By": [
            0.0,
            754.12940596242360,
        ],
        # 'Bz' : [0.0, 8.68335830083499e-14, 8.109979554106818e-13, 2.2881775959027207e-12, 3.4172712318143877e-12, 4.161059825829489e-12]
    }

    for field in reference_data.keys():

        new_data = []

        for i, it in enumerate(range(0, 201, 200)):

            file = "{}_{:03d}.bin".format(field, it)

            (
                x_axis_name,
                x_data,
                y_axis_name,
                y_data,
                z_axis_name,
                z_data,
                data_name,
                data_map,
            ) = minipic_diag.read_3d_diag("diags/" + file)

            # print(len(x_data), len(y_data), len(z_data), np.shape(data_map))

            xv, yv, zv = np.meshgrid(x_data, y_data, z_data, indexing="ij")

            new_data.append(np.sum(np.abs(data_map) * xv * yv * zv))

        print("    - For field {}: {}".format(field, new_data))

        if evaluate:
            for i, it in enumerate(range(0, 201, 200)):
                if i == 0:
                    continue

                minipic_ci.evaluate(
                    new_data[i],
                    reference_data[field][i],
                    threshold,
                    "relative",
                    "{} field not similar".format(field),
                )
