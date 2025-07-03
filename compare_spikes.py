import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py
from pathlib import Path

from bmtk.utils.sonata.config import SonataConfig


parser = argparse.ArgumentParser()
parser.add_argument("--bmtk_json_path", nargs="+")
parser.add_argument("--canata-spikes-csv", nargs="?")
parser.add_argument("--names", default=None)
parser.add_argument("--save-as", default=None, type=str)


args = parser.parse_args()

# If not explicity defined, attempt to find canata spikes csv file
json_path = Path(args.bmtk_json_path[0])
if args.canata_spikes_csv is None:
    canata_spikes_path = Path(f"canata-{json_path.parent}/out/spikes.csv")

    if not canata_spikes_path.exists():
        print(f"... failed!!!! {canata_spikes_path} does not exists.")
    else:
        print(
            f'canata-spikes csv file not explicity set, attempting to use "{canata_spikes_path}"'
        )
else:
    canata_spikes_path = args.canata_spikes_csv

cfg = SonataConfig.from_json(json_path.as_posix())
spikes_path_bmtk = json_path.parent / cfg["output"]["spikes_file_csv"]

spikes_df_bmtk = pd.read_csv(spikes_path_bmtk, sep=" ")
spikes_df_arbor = pd.read_csv(canata_spikes_path)

add_spikes = []
add_configs = args.bmtk_json_path[1:]
for add_config in add_configs:
    add_json_path = Path(add_config)
    add_cfg = SonataConfig.from_json(add_json_path.as_posix())
    add_spikes_path = add_json_path.parent / add_cfg["output"]["spikes_file_csv"]
    add_spikes.append(pd.read_csv(add_spikes_path, sep=" "))

n_sims = 2 + len(add_spikes)

if args.names is None:
    names = ["bmtk"] * (n_sims - 1)
    names.append("arbor")
else:
    names = args.names.split(",")


populations = set(spikes_df_bmtk["population"].unique()) & set(
    spikes_df_arbor["population"].unique()
)
for pop in populations:
    print(pop)
    sub_spikes_bmtk = spikes_df_bmtk[spikes_df_bmtk["population"] == pop]

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

        labels_lu = {}
        if "pop_name" in node_types_df:
            for _, r in node_types_df.iterrows():
                # print(r)
                labels_lu[r["node_type_id"]] = f"{r['node_type_id']} ({r['pop_name']})"
            # label_names = node_types_df[['node_type_id', 'pop_name']].set_index('node_type_id')['pop_name']
        else:
            for _, r in node_types_df.iterrows():
                labels_lu[r["node_type_id"]] = r["node_type_id"]
            # pop_names_lu = node_types_df[['node_type_id', 'pop_name']].set_index('node_type_id')['pop_name']

        sub_spikes_bmtk = pd.merge(sub_spikes_bmtk, nodes_df, how="left", on="node_ids")
        break

    sub_spikes_arbor = spikes_df_arbor[spikes_df_arbor["population"] == pop]
    sub_spikes_arbor = pd.merge(
        sub_spikes_arbor, nodes_df, how="left", left_on="gid", right_on="node_ids"
    )

    tmin = 0.0
    tmax = (
        np.ceil(
            np.max(
                [sub_spikes_bmtk["timestamps"].max(), sub_spikes_arbor["time"].max()]
            )
            / 1000.0
        )
        * 1000.0
    )
    sim_time_s = (tmax - tmin) / 1000.0

    fig, ax = plt.subplots(n_sims, 1, figsize=(10, 4 + n_sims))
    print("boo")
    if "node_type_id" in sub_spikes_bmtk.columns:
        print("HERE")
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color_map = {
            nid: colors[idx % len(colors)]
            for idx, nid in enumerate(nodes_df["node_type_id"].unique())
        }
        node_counts_df = nodes_df.groupby(["node_type_id", "model_type"])[
            "node_ids"
        ].agg("count")

        firing_rates = {
            nid: np.zeros(n_sims) for nid in nodes_df["node_type_id"].unique()
        }
        sems = {nid: np.zeros(n_sims) for nid in nodes_df["node_type_id"].unique()}

        ax_idx = 0
        for (typeid, modeltype), grp_df in sub_spikes_bmtk.groupby(
            ["node_type_id", "model_type"]
        ):
            n_nodes = node_counts_df[typeid, modeltype]
            print("HERE")
            ax[0].scatter(
                grp_df["timestamps"],
                grp_df["node_ids"],
                s=5,
                label=typeid,
                marker="o" if modeltype == "biophysical" else "v",
                color=color_map[typeid],
            )
            # print(f'{typeid} > {len(grp_df)} > {n_nodes}')
            firing_rates[typeid][0] = len(grp_df) / n_nodes / sim_time_s
            if n_nodes > 1:
                std = grp_df["node_ids"].value_counts().std()
                sems[typeid][ax_idx] = std / np.sqrt(
                    len(grp_df["node_ids"].value_counts())
                )

        for add_spikes_df in add_spikes:
            ax_idx += 1
            add_spikes_subpop = add_spikes_df[add_spikes_df["population"] == pop]
            add_spikes_subpop = pd.merge(
                add_spikes_subpop, nodes_df, how="left", on="node_ids"
            )

            for (typeid, modeltype), grp_df in add_spikes_subpop.groupby(
                ["node_type_id", "model_type"]
            ):
                n_nodes = node_counts_df[typeid, modeltype]
                ax[ax_idx].scatter(
                    grp_df["timestamps"],
                    grp_df["node_ids"],
                    s=5,
                    label=typeid,
                    marker="o" if modeltype == "biophysical" else "v",
                    color=color_map[typeid],
                )
                firing_rates[typeid][ax_idx] = len(grp_df) / n_nodes / sim_time_s
                if n_nodes > 1:
                    std = grp_df["node_ids"].value_counts().std()
                    sems[typeid][ax_idx] = std / np.sqrt(
                        len(grp_df["node_ids"].value_counts())
                    )

        ax_idx += 1
        for (typeid, modeltype), grp_df in sub_spikes_arbor.groupby(
            ["node_type_id", "model_type"]
        ):
            n_nodes = node_counts_df[typeid, modeltype]
            ax[ax_idx].scatter(
                grp_df["time"],
                grp_df["gid"],
                s=5,
                label=typeid,
                marker="o" if modeltype == "biophysical" else "v",
                color=color_map[typeid],
            )
            firing_rates[typeid][ax_idx] = len(grp_df) / n_nodes / sim_time_s
            if n_nodes > 1:
                std = grp_df["node_ids"].value_counts().std()
                sems[typeid][ax_idx] = std / np.sqrt(
                    len(grp_df["node_ids"].value_counts())
                )

        # for i in range(ax_idx+1):
        #     ax[i].legend(loc='upper right', markerscale=1, bbox_to_anchor=(1.1, 1.05))
        ax[0].legend(loc="upper right", markerscale=1, bbox_to_anchor=(1.1, 1.05))
    else:
        ax[0].scatter(sub_spikes_bmtk["timestamps"], sub_spikes_bmtk["node_ids"])
        ax[1].scatter(sub_spikes_arbor["time"], sub_spikes_arbor["gid"])
        firing_rates = None

    # Make sure both plots have same x-axis
    for i in range(ax_idx + 1):
        ax[i].set_xlim([tmin, tmax])

    # Make sure both plots have same y-axis
    ymin = -1
    ymax = (
        np.max([sub_spikes_bmtk["node_ids"].max(), sub_spikes_arbor["gid"].max()]) + 1
    )
    for i in range(ax_idx + 1):
        ax[i].set_ylim([ymin, ymax])

    for i in range(ax_idx + 1):
        ax[i].set_ylabel(names[i])

    # plt.tight_layout()
    if args.save_as is not None:
        os.makedirs("figs", exist_ok=True)
        fig.savefig(f"figs/raster.{args.save_as}.png", dpi=300, bbox_inches="tight")

    if firing_rates is not None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4 + n_sims))
        pos_beg = 1
        labels = []
        label_positions = []
        for name, frs in firing_rates.items():
            pos_end = pos_beg + len(frs)
            positions = np.arange(pos_beg, pos_end)
            ax.bar(
                positions,
                frs,
                color=["b", "r", "g", "k", "y"],
                label=names if pos_beg == 1 else None,
            )
            ax.errorbar(positions, frs, yerr=sems[name], fmt=".")
            labels.append(labels_lu[name])
            # labels.append(name)
            label_positions.append(pos_beg + (n_sims - 1) / 2)
            pos_beg = pos_end + 1

        ax.set_xticks(label_positions)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_ylabel("firing rate (Hz)")
        ax.legend()

        # plt.tight_layout()
        if args.save_as is not None:
            fig.savefig(
                f"figs/firing_rates.{args.save_as}.png", dpi=300, bbox_inches="tight"
            )


# pd.set_option('display.max_rows', None)
# print(spikes_df_arbor.value_counts(['gid', 'population']).sort_index())

# from pprint import pprint

# canata_out_dir = Path(canata_spikes_path).parent
# fig, ax = plt.subplots(15, 1)

# for report_name, report in cfg['reports'].items():
#     if report['module'] == 'membrane_report':
#         # print(cfg['output']['output_dir'])
#         report_path = json_path.parent / cfg['output']['output_dir'] / f'{report_name}.h5'
#         if report_path.exists():
#             with h5py.File(report_path, 'r') as h5:
#                 data = h5[f'/report/{pop}/data']
#                 mapping_grp = h5[f'/report/{pop}/mapping']
#                 times = np.arange(mapping_grp['time'][0], mapping_grp['time'][1], step=mapping_grp['time'][2])
#                 node_ids = mapping_grp['node_ids'][()]

#                 for idx, node_id in enumerate(node_ids):
#                     ax[idx].plot(times, data[:, idx])

#                     arbor_trace_path = canata_out_dir / f'gid_{node_id}-tag_0.csv'
#                     if arbor_trace_path.exists():
#                         print('HERE')
#                         arbor_trace_df = pd.read_csv(arbor_trace_path, sep=',')
#                         times_arbor = arbor_trace_df.iloc[:, 0]
#                         traces_arbor = arbor_trace_df.iloc[:, 1]
#                         ax[idx].plot(times_arbor, traces_arbor)
#                         # break


# # exit()


plt.tight_layout()
plt.show()
