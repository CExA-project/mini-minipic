"""Validation script for the `default` setup."""

import os

import numpy as np

from libminipic import ci as minipic_ci
from libminipic import diag as minipic_diag
from libminipic.exceptions import MissingFileMiniPICError
from libminipic.validate import THRESHOLD


def validate(evaluate=True, threshold=THRESHOLD):

    # Get the name of the current file (without extension) to use as a label for the validation
    validation_label = os.path.basename(__file__).split(".")[0]

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
        for it in range(0, 201, 200):
            file = "{}_{:03d}.vtk".format(field, it)
            output_file_list.append(file)

    # Add *.bin files
    for field in ["cloud_s00", "cloud_s01", "diag_w_gamma_s00", "diag_w_gamma_s01"]:
        for it in range(0, 201, 200):
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
    # Reference database

    references =  {
        "species_initial_scalars": [
            [
                0,
                14155776.0,
                1.499884908404337e-07
            ],
            [
                0,
                14155776.0,
                1.500028565897728e-07
            ]
        ],
        "species_final_scalars": [
            [
                200,
                14155776.0,
                1.499884907814349e-07
            ],
            [
                200,
                14155776.0,
                1.50002856588209e-07
            ]
        ],
        "fields_final": [
            200,
            2.373956450314232e-17,
            2.401477383433454e-17,
            2.279647260485128e-17,
            2.868246065305342e-18,
            2.445340639904533e-18,
            4.07796581368127e-18
        ],
        "gamma_spectrum_sum": [
            [
                1.0150531572334735e-05
            ],
            [
                1.0001577013349268e-05
            ]
        ],
        "cloud_initial": [
            [
                7077889.707407768,
                7077888.009019216,
                7077886.385895036,
                1135055.8230955107,
                1135055.584002778,
                1135050.7436878507
            ],
            [
                7077889.707407768,
                7077888.009019216,
                7077886.385895036,
                26358.735090712158,
                26359.07972500561,
                26359.00342069896
            ]
        ],
        "cloud_final": [
            [
                7077799.296687735,
                7078373.593529964,
                7078337.84302292,
                1135055.8229151054,
                1135055.5837926304,
                1135050.7435688768
            ],
            [
                7075627.918776955,
                7077880.4750964185,
                7077944.363062884,
                26358.735090630078,
                26359.07972484865,
                26359.003420551046
            ]
        ]
    }

    # ______________________________________________________________________
    # Check scalars

    print(" > Checking scalars")

    # Check initial scalar for species

    reference_data = references["species_initial_scalars"]

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
        else:
            references["species_initial_scalars"][ispecies] = [iteration, particles, energy]

    # Check final scalar for species

    reference_data = references["species_final_scalars"]

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
        else:
            references["species_final_scalars"][ispecies] = [iteration, particles, energy]

    # Check final scalar values for fields

    reference_data = references["fields_final"]

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
    else:
        references["fields_final"] = [iteration, Ex, Ey, Ez, Bx, By, Bz]

    # ______________________________________________________________________
    # Check gamma spectrums

    reference_sum_data = references["gamma_spectrum_sum"]

    print(" > Checking gamma spectrums")

    for ispecies in range(2):

        new_data = []

        for i, it in enumerate(range(0, 200, 200)):

            file = "diag_w_gamma_s{:02d}_{:03d}.bin".format(ispecies, it)

            x_axis_name, x_min, x_max, x_data, data_name, data = (
                minipic_diag.read_1d_diag("diags/" + file)
            )

            new_data.append(float(np.sum(np.abs(data * x_data))))

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
        else:
            references["gamma_spectrum_sum"][ispecies] = new_data

    # ______________________________________________________________________
    # Check initial cloud file (particle initialization)

    reference_data = references["cloud_initial"]

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
        else:
            references["cloud_initial"][ispecies] = [x_sum, y_sum, z_sum, px_sum, py_sum, pz_sum]

    # ______________________________________________________________________
    # Check final cloud file (particle initialization)

    reference_data = references["cloud_final"]

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
        else:
            references["cloud_final"][ispecies] = [x_sum, y_sum, z_sum, px_sum, py_sum, pz_sum]

    # ______________________________________________________________________
    # Write updated references to JSON file if not evaluating

    if not evaluate:
        import json

        json_file_name = validation_label

        with open(json_file_name, "w") as f:
            json.dump(references, f, indent=4)
        print(f" > New references written to {json_file_name}")

        print(f" > New references: \n {json.dumps(references, indent=4)}")

