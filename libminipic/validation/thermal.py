"""Validation script for the `default` setup."""

import os

import numpy as np

from libminipic import ci as minipic_ci
from libminipic import diag as minipic_diag
from libminipic.exceptions import MissingFileMiniPICError
from libminipic.validate import THRESHOLD


def validate(evaluate=True, threshold=THRESHOLD):
    """
    Validate the `thermal` setup.
    """

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
        for it in range(0, 300, 50):
            file = "{}_{:03d}.vtk".format(field, it)
            output_file_list.append(file)

    # Add *.bin files
    for field in ["cloud_s00", "cloud_s01", "diag_w_gamma_s00", "diag_w_gamma_s01"]:
        for it in range(0, 300, 50):
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

    references = {
        "species_initial_scalars": [
            [
                0,
                262144.0,
                1.499802569489301e-07
            ],
            [
                0,
                262144.0,
                1.500982120183951e-07
            ]
        ],
        "species_final_scalars": [
            [
                300,
                262144.0,
                1.499801677227092e-07
            ],
            [
                300,
                262144.0,
                1.500981415474697e-07
            ]
        ],
        "fields_final": [
            300,
            8.707252968548583e-16,
            8.187202771374718e-16,
            8.133778516563601e-16,
            1.014322708204903e-15,
            2.738517219101238e-15,
            2.932533327560371e-15
        ],
        "gamma_spectrum_sum": [
            [
                1.0150306498218023e-05,
                1.0150305070953287e-05,
                1.015031026575343e-05,
                1.0150307734880672e-05,
                1.0150311192601487e-05,
                1.0150308434741582e-05
            ],
            [
                1.000157589162598e-05,
                1.0001575891523015e-05,
                1.000157589136796e-05,
                1.000157589118226e-05,
                1.0001575890954698e-05,
                1.0001575890766535e-05
            ]
        ],
        "cloud_initial": [
            [
                131076.16027212347,
                131072.115340914,
                131069.5246495383,
                21014.376991466852,
                21028.564131718493,
                21014.480007866434
            ],
            [
                131076.16027212347,
                131072.115340914,
                131069.5246495383,
                488.339374560899,
                488.375137441201,
                488.2761036710392
            ]
        ],
        "cloud_final": [
            [
                131059.99739212362,
                131023.15182992321,
                131142.1355539146,
                21014.42145328788,
                21028.61699863085,
                21014.45806797341
            ],
            [
                131100.95742997798,
                130988.17625004012,
                131067.20585065623,
                488.34037148012436,
                488.373555886603,
                488.2763737032678
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

        for i, it in enumerate(range(0, 300, 50)):

            file = "diag_w_gamma_s{:02d}_{:03d}.bin".format(ispecies, it)

            x_axis_name, x_min, x_max, x_data, data_name, data = (
                minipic_diag.read_1d_diag("diags/" + file)
            )

            new_data.append(float(np.sum(np.abs(data * x_data))))

        print("    - For species {}: {}".format(ispecies, new_data))

        if evaluate:
            for i, it in enumerate(range(0, 300, 50)):
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

        file = "cloud_s{:02}_300.bin".format(ispecies)

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

        print(
            " > New references: \n {}".format(json.dumps(references, indent=4))
        )

