from pathlib import Path
import argparse
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bmtk.utils.sonata.config import SonataConfig


parser = argparse.ArgumentParser()
parser.add_argument(
    "--bmtk_json_path",
    # nargs=1,
    default="config.simulation.aibs_axon.syns.json",
)
parser.add_argument(
    "--canata_output_dir",
    # nargs=1,
    default="cantata-sim/out",
)
parser.add_argument(
    "--bmtk_json_path2",
    default=None,
    help="Optional second configuration file to compare as subplot",
)
parser.add_argument(
    "--canata_output_dir2",
    default=None,
    help="Output directory for second configuration",
)
parser.add_argument(
    "--node_types",
    default="network_axon/v1_node_types.csv",
    type=str,
)
parser.add_argument("--save-dir", default="figs", type=str)
parser.add_argument("--save-as", default="biocells-syns", type=str)
parser.add_argument("--show", action="store_true")
parser.add_argument("--cells", default=None, type=str)


args = parser.parse_args()

json_path = Path(args.bmtk_json_path)
canata_out_dir = Path(args.canata_output_dir)

# Check if second configuration is provided
json_path2 = None
canata_out_dir2 = None
cfg2 = None
if args.bmtk_json_path2:
    json_path2 = Path(args.bmtk_json_path2)
    canata_out_dir2 = Path(args.canata_output_dir2) if args.canata_output_dir2 else None
    cfg2 = SonataConfig.from_json(json_path2.as_posix())

cfg = SonataConfig.from_json(json_path.as_posix())

cell_list = None
if args.cells:
    cell_list = [int(cid) for cid in args.cells.split(",")]


def get_nodes_labels_lu(pop, config, json_path_ref):
    for nodes_dict in config.nodes:
        nodes_path = json_path_ref.parent / nodes_dict["nodes_file"]
        h5file = h5py.File(nodes_path, "r")
        if pop not in h5file["/nodes"]:
            continue

        node_types_df = pd.read_csv(
            json_path_ref.parent / nodes_dict["node_types_file"], sep=" "
        )
        nodes_df = pd.DataFrame(
            {
                "node_ids": h5file[f"/nodes/{pop}/node_id"][()],
                "node_type_id": h5file[f"/nodes/{pop}/node_type_id"][()],
            }
        )

        subtypes_df = (
            node_types_df
            if "population" not in node_types_df.columns
            else node_types_df[node_types_df["population"] == pop]
        )
        nodes_df = pd.merge(nodes_df, subtypes_df, how="left", on="node_type_id")

        nodes_df["name"] = nodes_df.apply(
            lambda r: f"{r['node_type_id']} ({r['pop_name']})", axis=1
        )
        return nodes_df[["node_ids", "name"]].set_index("node_ids")

        # nodes nodes_df[['node_ids', 'node_type_id', 'pop_name']].set_index(node_ids)


def load_trace_data(config, json_path_ref, canata_out_dir_ref, node_types, node_id):
    """Load BMTK and Arbor trace data for a given node_id"""
    for report_name, report in config["reports"].items():
        if report["module"] == "membrane_report":
            report_path = (
                json_path_ref.parent
                / config["output"]["output_dir"]
                / f"{report_name}.h5"
            )

            if report_path.exists():
                with h5py.File(report_path, "r") as h5:
                    for pop in h5["/report"].keys():
                        data = h5[f"/report/{pop}/data"]
                        mapping_grp = h5[f"/report/{pop}/mapping"]
                        times = np.arange(
                            mapping_grp["time"][0],
                            mapping_grp["time"][1],
                            step=mapping_grp["time"][2],
                        )
                        node_ids = mapping_grp["node_ids"][()]

                        if node_id in node_ids:
                            idx = np.where(node_ids == node_id)[0][0]
                            bmtk_times = times
                            bmtk_trace = data[:, idx]

                            # Load Arbor trace if available
                            arbor_times = None
                            arbor_trace = None
                            arbor_trace_path = (
                                canata_out_dir_ref / f"gid_{node_id}-tag_0.csv"
                            )
                            if arbor_trace_path.exists():
                                arbor_trace_df = pd.read_csv(arbor_trace_path, sep=",")
                                arbor_times = arbor_trace_df.iloc[:, 0]
                                arbor_trace = arbor_trace_df.iloc[:, 1]

                            # Get mechanism info
                            mechanism = ""
                            try:
                                # Find the node_type_id for this node_id
                                node_type_id = h5[f"/nodes/{pop}/node_type_id"][()][idx]
                                mechanism = (
                                    node_types.loc[
                                        node_types["node_type_id"] == node_type_id,
                                        "dynamics_params",
                                    ]
                                    .values[0]
                                    .split(".")[0]
                                    .split("_fit")[-1]
                                )
                            except:
                                mechanism = ""

                            return {
                                "bmtk_times": bmtk_times,
                                "bmtk_trace": bmtk_trace,
                                "arbor_times": arbor_times,
                                "arbor_trace": arbor_trace,
                                "mechanism": mechanism,
                                "pop": pop,
                            }
    return None


print("| Tag | Stim Type | RMS Error | Delta | Arbor | BMTK |")

# Load node types once
node_types = pd.read_csv(args.node_types, sep=" ")

# Get all node_ids to process from the first configuration
all_node_ids = set()
for report_name, report in cfg["reports"].items():
    if report["module"] == "membrane_report":
        report_path = (
            json_path.parent / cfg["output"]["output_dir"] / f"{report_name}.h5"
        )
        if report_path.exists():
            with h5py.File(report_path, "r") as h5:
                for pop in h5["/report"].keys():
                    mapping_grp = h5[f"/report/{pop}/mapping"]
                    node_ids = mapping_grp["node_ids"][()]
                    all_node_ids.update(node_ids)

# Process each node_id

errors_df = pd.DataFrame(columns=["node_id", "stimulus_type", "rms_error"])
delta_df = pd.DataFrame(columns=["node_id", "stimulus_type", "delta_error"])

for node_id in all_node_ids:
    if cell_list and node_id not in cell_list:
        continue

    # Load data from first configuration
    data1 = load_trace_data(cfg, json_path, canata_out_dir, node_types, node_id)
    if data1 is None:
        continue

    # Load data from second configuration if provided
    data2 = None
    if cfg2 and canata_out_dir2:
        data2 = load_trace_data(cfg2, json_path2, canata_out_dir2, node_types, node_id)

    # Determine subplot layout
    n_subplots = 2 if data2 else 1
    fig, axes = plt.subplots(n_subplots, 1, figsize=(15, 4 * n_subplots))
    if n_subplots == 1:
        axes = [axes]  # Make it iterable

    # Get node label
    labels_lu = get_nodes_labels_lu(data1["pop"], cfg, json_path)
    node_label = (
        labels_lu.loc[node_id]["name"]
        if node_id in labels_lu.index
        else f"Node {node_id}"
    )

    # Plot first configuration
    ax1 = axes[0]
    ax1.plot(data1["bmtk_times"], data1["bmtk_trace"], label="bmtk (config 1)")
    if data1["arbor_trace"] is not None:
        ax1.plot(data1["arbor_times"], data1["arbor_trace"], label="arbor (config 1)")

        # Calculate and print error metrics for config 1
        delta = np.sum((data1["arbor_trace"].values - data1["bmtk_trace"]) ** 2)
        total_arbor = np.sum(data1["arbor_trace"].values ** 2)
        total_bmtk = np.sum(data1["bmtk_trace"] ** 2)
        max_bmtk = np.max(np.abs(data1["bmtk_trace"]))
        if (
            (delta / total_arbor > 0.001 or np.isnan(total_arbor))
            and not np.isnan(max_bmtk)
            and max_bmtk < 200
        ):
            print(
                f"| {node_id:>10}{data1['mechanism']:<10} | Syns | {delta/total_arbor:.3f} | {delta:.3f} | {total_arbor:.3f}|  {total_bmtk:.3f} |"
            )

        errors_df = pd.concat(
            [
                errors_df,
                pd.DataFrame(
                    {
                        "node_id": [node_id],
                        "stimulus_type": ["Syn Inputs"],
                        "rms_error": [np.sqrt(delta / len(data1["bmtk_trace"]))],
                    }
                ),
            ],
            ignore_index=True,
        )

        delta_df = pd.concat(
            [
                delta_df,
                pd.DataFrame(
                    {
                        "node_id": [node_id],
                        "stimulus_type": ["Syn Inputs"],
                        "delta_error": [delta / total_arbor],
                    }
                ),
            ],
            ignore_index=True,
        )

    ax1.set_ylabel("mV")
    ax1.set_xlabel("ms")
    ax1.legend(loc="upper right")
    config1_name = json_path.stem
    ax1.set_title(f"gid_{node_id}-{node_label}{data1['mechanism']} - {config1_name}")

    # Plot second configuration if available
    if data2:
        ax2 = axes[1]
        ax2.plot(
            data2["bmtk_times"],
            data2["bmtk_trace"],
            label="bmtk (config 2)",
            # color="orange",
        )
        if data2["arbor_trace"] is not None:
            ax2.plot(
                data2["arbor_times"],
                data2["arbor_trace"],
                label="arbor (config 2)",
                # color="red",
            )
            # Calculate and print error metrics for config 1
            delta2 = np.sum((data2["arbor_trace"].values - data2["bmtk_trace"]) ** 2)
            total_arbor2 = np.sum(data2["arbor_trace"].values ** 2)
            total_bmtk2 = np.sum(data2["bmtk_trace"] ** 2)
            max_bmtk2 = np.max(np.abs(data2["bmtk_trace"]))
            if (
                (delta2 / total_arbor2 > 0.001 or np.isnan(total_arbor2))
                and not np.isnan(max_bmtk2)
                and max_bmtk2 < 200
            ):
                print(
                    f"| {node_id:>10}{data2['mechanism']:<10} | iClamp | {delta2/total_arbor2:.3f} | {delta2:.3f} | {total_arbor2:.3f}|  {total_bmtk2:.3f} |"
                )

            errors_df = pd.concat(
                [
                    errors_df,
                    pd.DataFrame(
                        {
                            "node_id": [node_id],
                            "stimulus_type": ["iClamp"],
                            "rms_error": [np.sqrt(delta2 / len(data2["bmtk_trace"]))],
                        }
                    ),
                ],
                ignore_index=True,
            )
            delta_df = pd.concat(
                [
                    delta_df,
                    pd.DataFrame(
                        {
                            "node_id": [node_id],
                            "stimulus_type": ["iClamp"],
                            "delta_error": [
                                delta2 / np.sum(data2["arbor_trace"].values ** 2)
                            ],
                        }
                    ),
                ],
                ignore_index=True,
            )

        ax2.set_ylabel("mV")
        ax2.set_xlabel("ms")
        ax2.legend(loc="upper right")
        config2_name = json_path2.stem
        ax2.set_title(
            f"gid_{node_id}-{node_label}{data2['mechanism']} - {config2_name}"
        )

    # Save plot
    if args.save_as is not None:
        suffix = "_comparison" if data2 else ""
        save_path = f"trace.{node_id}{data1['mechanism']}.{args.save_as}{suffix}.pdf"
        plt.tight_layout()
        os.makedirs(args.save_dir, exist_ok=True)
        fig.savefig(os.path.join(args.save_dir, save_path))

        # errors_df.to_csv(
        #     os.path.join(args.save_dir, f"rms_errors.{args.save_as}.csv"), index=False
        # )
        delta_df.to_csv(
            os.path.join(args.save_dir, f"delta_errors.{args.save_as}.csv"), index=False
        )

    plt.close(fig)

if args.show:
    plt.tight_layout()
    plt.show()
