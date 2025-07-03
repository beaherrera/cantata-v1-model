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

cfg = SonataConfig.from_json(json_path.as_posix())

cell_list = None
if args.cells:
    cell_list = [int(cid) for cid in args.cells.split(",")]


def get_nodes_labels_lu(pop):
    for nodes_dict in cfg.nodes:
        nodes_path = json_path.parent / nodes_dict["nodes_file"]
        h5file = h5py.File(nodes_path, "r")
        if pop not in h5file["/nodes"]:
            continue

        node_types_df = pd.read_csv(
            json_path.parent / nodes_dict["node_types_file"], sep=" "
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


print("| Tag | RMS Error | Delta | Arbor | BMTK |")
for report_name, report in cfg["reports"].items():
    if report["module"] == "membrane_report":
        # print(cfg['output']['output_dir'])
        report_path = (
            json_path.parent / cfg["output"]["output_dir"] / f"{report_name}.h5"
        )

        node_types = pd.read_csv(args.node_types, sep=" ")

        if report_path.exists():
            with h5py.File(report_path, "r") as h5:
                for pop in h5["/report"].keys():
                    labels_lu = get_nodes_labels_lu(pop)

                    data = h5[f"/report/{pop}/data"]
                    mapping_grp = h5[f"/report/{pop}/mapping"]
                    times = np.arange(
                        mapping_grp["time"][0],
                        mapping_grp["time"][1],
                        step=mapping_grp["time"][2],
                    )
                    node_ids = mapping_grp["node_ids"][()]
                    n_nodes = len(node_ids)

                    for idx, node_id in enumerate(node_ids):
                        if cell_list and node_id not in cell_list:
                            continue

                        # fig, ax = plt.subplots(n_nodes, 1, figsize=(15, 4 + n_nodes))
                        fig, ax = plt.subplots(1, 1, figsize=(15, 4))
                        ax.plot(times, data[:, idx], label="bmtk")
                        mechanism = (
                            node_types.loc[
                                node_types["node_type_id"].index.values == node_id,
                                "dynamics_params",
                            ]
                            .values[0]
                            .split(".")[0]
                            .split("_fit")[-1]
                        )

                        arbor_trace_path = canata_out_dir / f"gid_{node_id}-tag_0.csv"
                        if arbor_trace_path.exists():
                            arbor_trace_df = pd.read_csv(arbor_trace_path, sep=",")
                            times_arbor = arbor_trace_df.iloc[:, 0]
                            traces_arbor = arbor_trace_df.iloc[:, 1]
                            delta = np.sum((traces_arbor.values - data[:, idx]) ** 2)
                            total_arbor = np.sum(traces_arbor.values**2)
                            total_bmtk = np.sum(data[:, idx] ** 2)
                            max_bmtk = np.max(np.abs(data[:, idx]))
                            if (
                                (delta / total_arbor > 0.001 or np.isnan(total_arbor))
                                and not np.isnan(max_bmtk)
                                and max_bmtk < 200
                            ):
                                print(
                                    f"| {node_id:>10}{mechanism:<10} | {delta/total_arbor:.3f} | {delta:.3f} | {total_arbor:.3f}|  {total_bmtk:.3f} |"
                                )
                            ax.plot(times_arbor, traces_arbor, label="arbor")

                        ax.set_ylabel("mV")
                        ax.set_xlabel("ms")
                        ax.legend(loc="upper right")
                        ax.set_title(labels_lu.loc[node_id]["name"] + mechanism)
                        if args.save_as is not None:
                            save_path = f"trace.{node_id}{mechanism}.{args.save_as}.pdf"
                            plt.tight_layout()
                            os.makedirs(args.save_dir, exist_ok=True)
                            fig.savefig(os.path.join(args.save_dir, save_path))
                        plt.close(fig)

if args.show:
    plt.tight_layout()
    plt.show()
