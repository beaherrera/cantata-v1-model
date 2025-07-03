import os
import sys
import pandas as pd
import json
import argparse
import logging

# from mpi4py import MPI
from pathlib import Path
import numpy as np
import ast

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()
except ImportError as ie:
    mpi_rank = 0
    mpi_size = 1


from bmtk.builder import NetworkBuilder
from bmtk.builder.bionet import rand_syn_locations
from bmtk.utils import sonata

from build_files.node_funcs import (
    generate_random_positions,
    generate_positions_grids,
    get_filter_temporal_params,
    get_filter_spatial_size,
)
from build_files.edge_funcs import (
    select_lgn_sources,
    compute_pair_type_parameters,
    connect_cells,
)


logger = logging.getLogger()


def add_nodes_v1(
    node_props_path="build_files/biophys_props/v1_node_models.json",
    fraction=1.0,
    rng_seed=None,
):
    if rng_seed is not None:
        np.random.seed(rng_seed)

    v1_models = json.load(open(node_props_path, "r"))
    inner_radial_range = v1_models["inner_radial_range"]
    outer_radial_range = v1_models["outer_radial_range"]

    net = NetworkBuilder("v1")  # , adaptor_cls="V05")
    for location, loc_dict in v1_models["locations"].items():
        for pop_name, pop_dict in loc_dict.items():
            for model_props in pop_dict["models"]:
                N = int(np.max((model_props["N"] * fraction, 1)))
                model_type = model_props["model_type"]

                if model_type != "biophysical":
                    continue

                radial_range = (
                    inner_radial_range
                    if model_type == "biophysical"
                    else outer_radial_range
                )
                depth_range = pop_dict["depth_range"]
                positions = generate_random_positions(N, depth_range, radial_range)
                tuning_angle = np.linspace(0.0, 360.0, N, endpoint=False)

                net.add_nodes(
                    N=N,
                    rotation_angle_xaxis=np.zeros(N),
                    rotation_angle_yaxis=(
                        2 * np.pi * np.random.random(N)
                        if model_type == "biophysical"
                        else np.zeros(N)
                    ),
                    rotation_angle_zaxis=np.full(
                        N, model_props.get("rotation_angle_zaxis", 0.0)
                    ),
                    x=positions[:, 0],
                    y=positions[:, 1],
                    z=positions[:, 2],
                    tuning_angle=tuning_angle,
                    node_type_id=model_props["node_type_id"],
                    model_type=model_props["model_type"],
                    model_template=model_props["model_template"],
                    dynamics_params=model_props["dynamics_params"],
                    ei=pop_dict["ei"],
                    location=location,
                    pop_name=("LIF" if model_type != "biophysical" else "") + pop_name,
                    population="v1",
                    model_processing=(
                        "aibs_axon" if model_type == "biophysical" else None
                    ),  # model_props.get('model_processing', None),
                    morphology=model_props.get("morphology", None),
                )

    return net


def add_nodes_lgn(
    node_props_path="build_files/biophys_props/lgn_node_models.json", rng_seed=None
):
    if rng_seed is not None:
        np.random.seed(rng_seed)

    lgn_models = json.load(open(node_props_path, "r"))
    X_grids = 15
    Y_grids = 10
    X_len = 240.0
    Y_len = 120.0

    net = NetworkBuilder("lgn")
    for model, params in lgn_models.items():
        pop_name = params["model_id"]
        pop_N = params["N"] * X_grids * Y_grids

        positions = generate_positions_grids(
            params["N"], X_grids, Y_grids, X_len, Y_len
        )

        # Get spatial filter size of cells
        filter_sizes = get_filter_spatial_size(
            params["N"], X_grids, Y_grids, params["size_range"]
        )

        # Get filter temporal parameters
        filter_params = get_filter_temporal_params(
            params["N"], X_grids, Y_grids, pop_name
        )

        net.add_nodes(
            N=pop_N,
            x=positions[:, 0],
            y=positions[:, 1],
            spatial_size=filter_sizes,
            kpeaks_dom_0=filter_params[:, 0],
            kpeaks_dom_1=filter_params[:, 1],
            weight_dom_0=filter_params[:, 2],
            weight_dom_1=filter_params[:, 3],
            delay_dom_0=filter_params[:, 4],
            delay_dom_1=filter_params[:, 5],
            kpeaks_non_dom_0=filter_params[:, 6],
            kpeaks_non_dom_1=filter_params[:, 7],
            weight_non_dom_0=filter_params[:, 8],
            weight_non_dom_1=filter_params[:, 9],
            delay_non_dom_0=filter_params[:, 10],
            delay_non_dom_1=filter_params[:, 11],
            tuning_angle=filter_params[:, 12],
            sf_sep=filter_params[:, 13],
            pop_name=pop_name,
            level_of_detail="filter",
            ei="e",
            location="LGN",
            model_type="virtual",
        )

    return net


def add_nodes_bkg():
    bkg = NetworkBuilder("bkg")
    bkg.add_nodes(
        N=1,
        pop_name="SG_001",
        ei="e",
        location="BKG",
        model_type="virtual",
        x=[-91.23767151810344],
        y=[233.43548226294524],
    )
    return bkg


def find_direction_rule(src_label, trg_label):
    src_ei = "e" if src_label.startswith("e") or src_label.startswith("LIFe") else "i"
    trg_ei = "e" if trg_label.startswith("e") or trg_label.startswith("LIFe") else "i"

    if src_ei == "e" and trg_ei == "e":
        return "DirectionRule_EE", 30.0

    elif src_ei == "e" and trg_ei == "i":
        return "DirectionRule_others", 90.0

    elif src_ei == "i" and trg_ei == "e":
        return "DirectionRule_others", 90.0

    else:
        return "DirectionRule_others", 50.0


def adjust_syn_weight(src, trg, syn_weight, direction_rule, weight_sigma):
    src_tuning = src["tuning_angle"]
    tar_tuning = trg["tuning_angle"]

    delta_tuning_180 = abs(abs((abs(tar_tuning - src_tuning) % 360.0) - 180.0) - 180.0)
    w_multiplier_180 = np.exp(-((delta_tuning_180 / weight_sigma) ** 2))

    if direction_rule == "DirectionRule_EE":
        x_src = src["x"]
        z_src = src["z"]

        x_tar = trg["x"]
        z_tar = trg["z"]

        delta_x = (x_tar - x_src) * 0.07
        delta_z = (z_tar - z_src) * 0.04

        theta_pref = tar_tuning * (np.pi / 180.0)
        xz = delta_x * np.cos(theta_pref) + delta_z * np.sin(theta_pref)
        sigma_phase = 1.0
        phase_scale_ratio = np.exp(-(xz**2 / (2 * sigma_phase**2)))
    else:
        phase_scale_ratio = 1.0

    return w_multiplier_180 * phase_scale_ratio * syn_weight


def add_v1_v1_edges(
    net,
    edge_props_path="build_files/biophys_props/v1_v1_edge_models.csv",
    rng_seed=None,
):
    if rng_seed is not None:
        np.random.seed(rng_seed)

    conn_weight_df = pd.read_csv(edge_props_path, sep=" ")

    for _, row in conn_weight_df.iterrows():
        node_type_id = row["target_model_id"]
        src_type = row["source_label"]
        trg_type = row["target_label"]
        src_trg_params = compute_pair_type_parameters(src_type, trg_type)

        weight_fnc, weight_sigma = find_direction_rule(src_type, trg_type)
        if src_trg_params["A_new"] > 0.0:
            if trg_type.startswith("LIF"):
                continue
            cm = net.add_edges(
                edge_type_id=row["edge_type_id"],
                source={"pop_name": src_type},
                target={"node_type_id": node_type_id},
                iterator="all_to_one",
                connection_rule=connect_cells,
                connection_params={"params": src_trg_params},
                dynamics_params=row["params_file"],
                model_template=None if trg_type.startswith("LIF") else "exp2syn",
                # syn_weight=row['weight_max'],
                delay=row["delay"],
                # weight_function=weight_fnc,
                # weight_sigma=weight_sigma
            )
            cm.add_properties(
                "syn_weight",
                rule=adjust_syn_weight,
                rule_params={
                    "syn_weight": row["weight_max"],
                    "direction_rule": weight_fnc,
                    "weight_sigma": weight_sigma,
                },
                dtypes=float,
            )

            # else:
            #     cm = net.add_edges(
            #         edge_type_id=row["edge_type_id"],
            #         source={"pop_name": src_type},
            #         target={"node_type_id": node_type_id},
            #         iterator="all_to_one",
            #         connection_rule=connect_cells,
            #         connection_params={"params": src_trg_params},
            #         dynamics_params=row["params_file"],
            #         model_template="exp2syn",
            #         # syn_weight=row['weight_max'],
            #         delay=row["delay"],
            #         # weight_function=weight_fnc,
            #         # weight_sigma=weight_sigma,
            #         # distance_range=row['distance_range'],
            #         # target_sections=row['target_sections']
            #     )

            if not trg_type.startswith("LIF"):
                # If the target node is biophysical, we need to add the synapse locations
                # and properties to the connection matrix.

                # Uses distance_range and target_sections to selection post-synaptic locations
                distance_range = (
                    ast.literal_eval(row["distance_range"])
                    if isinstance(row["distance_range"], str)
                    else row["distance_range"]
                )
                target_sections = (
                    ast.literal_eval(row["target_sections"])
                    if isinstance(row["target_sections"], str)
                    else row["target_sections"]
                )
                cm.add_properties(
                    [
                        "afferent_section_id",
                        "afferent_section_pos",
                        "afferent_swc_id",
                        "afferent_swc_pos",
                        "afferent_section_xcoords",
                        "afferent_section_ycoords",
                        "afferent_section_zcoords",
                    ],
                    # ['afferent_section_id', 'afferent_section_pos', 'afferent_swc_id', 'afferent_swc_pos'],
                    rule=rand_syn_locations,
                    rule_params={
                        "sections": target_sections,
                        "distance_range": distance_range,
                        # "return_swc": False,
                        "return_coords": False,
                        "morphology_dir": "./components/morphologies/axon_stubs",
                    },
                    dtypes=[int, float, int, float, float, float, float],
                    # dtypes=[int, float, int, float],
                )

                # cm.add_properties(
                #     "syn_weight",
                #     rule=adjust_syn_weight,
                #     rule_params={
                #         "syn_weight": row["weight_max"],
                #         "direction_rule": weight_fnc,
                #         "weight_sigma": weight_sigma,
                #     },
                #     dtypes=float,
                # )

    return net


def add_lgn_v1_edges(v1, lgn_net, x_len=240.0, y_len=120.0, rng_seed=None):
    if rng_seed is not None:
        np.random.seed(rng_seed)

    conn_weight_df = pd.read_csv(
        "build_files/biophys_props/lgn_v1_edge_models.csv", sep=" "
    )
    lgn_models = json.load(open("build_files/biophys_props/lgn_node_models.json", "r"))

    lgn_mean = (x_len / 2.0, y_len / 2.0)
    valid_target_node_types = {n["node_type_id"] for n in v1.nodes()}
    for _, row in conn_weight_df.iterrows():
        src_type = row["source_label"]
        trg_type = row["target_label"]
        target_node_type = row["target_model_id"]

        if target_node_type not in valid_target_node_types:
            continue

        if trg_type.startswith("LIF"):
            continue

        cm = lgn_net.add_edges(
            edge_type_id=row["edge_type_id"],
            source=lgn_net.nodes(location="LGN"),
            target=v1.nodes(node_type_id=target_node_type),
            iterator="all_to_one",
            connection_rule=select_lgn_sources,
            connection_params={
                "lgn_mean": lgn_mean,
                "lgn_models": lgn_models,
                "nsyns": row["nsyns"],
                "edge_type_id": row["edge_type_id"],
            },
            dynamics_params=row["params_file"],
            model_template=None if trg_type.startswith("LIF") else "exp2syn",
            # syn_weight=row['weight_max'],
            delay=row["delay"],
            # weight_function=row['weight_func'],
            # weight_sigma=row['weight_sigma']
        )
        cm.add_properties("syn_weight", rule=row["weight_max"], dtypes=float)

        if row["target_sections"] is not None:
            distance_range = (
                ast.literal_eval(row["distance_range"])
                if isinstance(row["distance_range"], str)
                else row["distance_range"]
            )
            target_sections = (
                ast.literal_eval(row["target_sections"])
                if isinstance(row["target_sections"], str)
                else row["target_sections"]
            )
            if not isinstance(distance_range, (list, tuple)):
                continue

            cm.add_properties(
                [
                    "afferent_section_id",
                    "afferent_section_pos",
                    "afferent_swc_id",
                    "afferent_swc_pos",
                    "afferent_section_xcoords",
                    "afferent_section_ycoords",
                    "afferent_section_zcoords",
                ],  # , 'afferent_swc_id', 'afferent_swc_pos'],
                rule=rand_syn_locations,
                rule_params={
                    "sections": target_sections,
                    "distance_range": distance_range,
                    # "return_swc": False,
                    "return_coords": False,
                    "morphology_dir": "./components/morphologies/axon_stubs",
                },
                dtypes=[int, float, int, float, float, float, float],  # , int, float],
            )
            # edge_params.update({
            #     'target_sections': row['target_sections'],
            #     'distance_range': row['distance_range']
            # })

        # lgn_net.add_edges(**edge_params)

    return lgn_net


def add_bkg_v1_edges(v1_net, bkg_net):
    conn_weight_df = pd.read_csv(
        "build_files/biophys_props/bkg_v1_edge_models.csv", sep=" "
    )

    for _, row in conn_weight_df.iterrows():
        trg_type = row["target_label"]
        target_node_type = row["target_model_id"]
        nsyns = row.get("nsyns", 1)

        if trg_type.startswith("LIF"):
            continue

        cm = bkg_net.add_edges(
            source=bkg_net.nodes(),
            target=v1_net.nodes(node_type_id=target_node_type),
            connection_rule=lambda s, t, n: n,
            connection_params={"n": nsyns},
            model_template=None if trg_type.startswith("LIF") else "exp2syn",
            dynamics_params=row["dynamics_params"],
            syn_weight=row["syn_weight"],
            delay=row["delay"],
        )
        # cm.add_properties("syn_weight", rule=row["syn_weight"], dtypes=float)

        if trg_type == "biophysical":
            distance_range = (
                ast.literal_eval(row["distance_range"])
                if isinstance(row["distance_range"], str)
                else row["distance_range"]
            )
            target_sections = (
                ast.literal_eval(row["target_sections"])
                if isinstance(row["target_sections"], str)
                else row["target_sections"]
            )

            cm.add_properties(
                [
                    "afferent_section_id",
                    "afferent_section_pos",
                    "afferent_swc_id",
                    "afferent_swc_pos",
                    "afferent_section_xcoords",
                    "afferent_section_ycoords",
                    "afferent_section_zcoords",
                ],  # , 'afferent_swc_id', 'afferent_swc_pos'],
                rule=rand_syn_locations,
                rule_params={
                    "sections": target_sections,
                    "distance_range": distance_range,
                    # "return_swc": False,
                    "return_coords": False,
                    "morphology_dir": "./components/morphologies/axon_stubs",
                },
                dtypes=[int, float, int, float, float, float, float],  # , int, float],
            )

        #     edge_params.update({
        #         # 'model_template': 'exp2syn',
        #         'target_sections': row['target_sections'],
        #         'distance_range': row['distance_range']
        #     })
        # bkg_net.add_edges(**edge_params)

    return bkg_net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the (biophysical) V1 (and lgn/background inputs) SONATA network files"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="network.full_axon",
        help="directory where network files will be saved.",
    )
    parser.add_argument(
        "--save-separate",
        action="store_true",
        default=False,
        help="Will enable v1, lgn, and bkg network files to be saved into separate "
        "sub-folders in the --output-dir folder.",
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="logs debugging info"
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=False,
        help="Will disable printing of output to console.",
    )
    parser.add_argument(
        "--log-file", type=str, default=None, help="log build process to a file."
    )
    parser.add_argument(
        "--rng-seed", type=int, default=100, help="seed number for random generator"
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Only build a fraction (0 - 1.0] of the total V1 nodes. default value: 1.0",
    )
    args, _ = parser.parse_known_args()

    # logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    log_file = f"logs/build_network.{args.fraction:.02f}frac.rank{mpi_rank:02d}of{mpi_size:02d}.txt"
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    log_format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    logger.setLevel(logging.DEBUG)

    fileHandler = logging.FileHandler(log_file, "w")
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    # log_handlers = [fileHandler]
    logger.addHandler(fileHandler)

    if mpi_rank == 0:
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        consoleHandler.setLevel(logging.INFO)
        logger.addHandler(consoleHandler)

    # if not args.quiet:
    #     console_handler = logging.StreamHandler(sys.stdout)
    #     console_handler.setFormatter(log_format)
    #     logger.addHandler(console_handler)
    # if args.log_file:
    #     file_logger = logging.FileHandler(args.log_file)
    #     file_logger.setFormatter(log_format)
    #     logger.addHandler(file_logger)

    v1_network_dir = os.path.join(args.output_dir, "v1" if args.save_separate else "")
    lgn_network_dir = os.path.join(args.output_dir, "lgn" if args.save_separate else "")
    bkg_network_dir = os.path.join(args.output_dir, "bkg" if args.save_separate else "")

    ## First Create the V1 network w/ recurrent connections.
    logger.info("Building v1 Network")
    v1 = add_nodes_v1(fraction=args.fraction, rng_seed=args.rng_seed)
    v1 = add_v1_v1_edges(v1, rng_seed=args.rng_seed)
    v1.build()
    logger.info(f'Saving v1 SONATA Network files to "{v1_network_dir}".')
    v1.save(output_dir=v1_network_dir)

    ## Once the V1 network has been created we can add feedforward inputs from the LGN.
    logger.info("Building lgn Network")
    lgn = add_nodes_lgn(rng_seed=args.rng_seed)
    lgn = add_lgn_v1_edges(v1, lgn, rng_seed=args.rng_seed)
    lgn.build()
    logger.info(f'Saving lgn SONATA Network files to "{lgn_network_dir}".')
    lgn.save(output_dir=lgn_network_dir)

    ## Finally add feedforward inputs from the background (BKG) onto the V1.
    logger.info("Building bkg Network")
    bkg = add_nodes_bkg()
    bkg = add_bkg_v1_edges(v1, bkg)
    bkg.build()
    logger.info(f'Saving bkg SONATA Network files to "{bkg_network_dir}".')
    bkg.save(output_dir=bkg_network_dir)
