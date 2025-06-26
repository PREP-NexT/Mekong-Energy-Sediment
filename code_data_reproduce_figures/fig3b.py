#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This script is used to generate figures for the manuscript."""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import pandas as pd
import numpy as np

def get_radius(data, max_v):
    """Get the radius of the pie chart based on the data and the max value.
    Use log scale to make sure the difference of size of the pie chart is not 
    too large.
    
    Refer to: `https://github.com/matplotlib/matplotlib/blob/
    1edae7bfc6b26e9db47dfffe13c86b6ab4ce0653/lib/matplotlib/
    axes/_axes.py#L3164`
    `xlim=(-1.25 + center[0], 1.25 + center[0])`
    `ylim=(-1.25 + center[1], 1.25 + center[1])`
    When we use pie chart, the radius is relative radius to 1.25
    times the minimum of the width and height of the axes.
    if set frame=True, the radius does not change.
    In summary, following the below code, the maximum radius over 
    figures is equal to the minimum of the width and height of 
    the axes.
    
    Parameters:
    -----------
    data: float
        The value of the current pie chart.
    max_v: float
        The max value of all pie charts over figures.
    """
    current_capacity = data
    max_capacity = max_v
    return np.sqrt(current_capacity/np.pi) / np.sqrt(max_capacity/np.pi) * 1.25

def map_data_to_size(data, radius_inch_max, max_v):
    """Map the data to the actual size of the pie chart.
    
    Parameters:
    -----------
    data: float
        The value of the current pie chart.
    radius_inch_max: float
        The actual size of the pie chart in inch while the data is the max 
        value.
    max_v: float
        The max value of all pie charts over figures.
    """
    radius = get_radius(data, max_v)
    return radius * radius_inch_max

def main(fig_format, df, name, legend_sizes, root_path):
    idx = pd.IndexSlice
    height = 70
    width = 90
    mm2inch = 1/25.4 # millimeter to inch
    df_fig2 = df.copy()
    solution_num = df_fig2.columns
    colors = ['#4F5357','#9A8479','#11758E','#FCB668','#709C80','#8DC53E','#934B76']
    tech = ['Coal', 'Gas', 'Hydro', 'Solar', 'Wind', 'Bioenergy', 'Li-ion']
    col_count = len(solution_num)
    row_count = len(tech)
    fig, axs = plt.subplots(
        row_count, col_count,
        figsize=(width*mm2inch,height*mm2inch),
        gridspec_kw={
            'left': 0.16, 'right': 1.0,
            'top': 0.9, 'bottom': 0.0
        }
    )
    # Get the max value total installed capacity during 2030-2050
    # across all solutions and all technologies to determine the
    # max size of the pie chart.
    max_v = df_fig2.groupby(level=[1]).sum().max(axis=1).max()
    for x_ix, (te, color) in enumerate(zip(tech, colors)):
        for y_ix, s in enumerate(solution_num):
            colormap = matplotlib.colors.LinearSegmentedColormap.from_list(
                'custom',
                [color, "#ffffff"],
                N=10
            )
            color_year = [colormap(0/10),  colormap(3/10), colormap(6/10)]
            data = df_fig2.loc[idx[:, te], s].values
            # add pie chart
            if data.sum() <= 0:
                axs[x_ix, y_ix].pie(
                    np.random.rand(data.shape[0]),
                    startangle=90,
                    radius=np.random.rand(),
                    colors=['none'] * len(color_year)
                )
            else:
                size = get_radius(data.sum(), max_v)
                axs[x_ix, y_ix].pie(
                    data, radius=size, startangle=90, colors=color_year,
                )
            # add labels
            if x_ix == 0:
                axs[x_ix, y_ix].text(
                    x=0.5, y=1.5, s=s, ha="center",va="center",
                    transform=axs[0, y_ix].transAxes
                )
            if y_ix == 0:
                axs[x_ix, y_ix].text(
                    x=-0.2, y=0.5, s=te, ha="right", va="center",
                    transform=axs[x_ix, 0].transAxes
                )
            # This part is to validate whether radius of pie chart is equal to
            # the minimum of the width and height of the axes.
            # ax = axs[x_ix, y_ix]
            # ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
            # ax.axvline(0, color='gray', linestyle='-', linewidth=0.5)
    # Get the mimimum value of the width and height of the axes. Unit is inch.
    # Use this value to set make legend size consistent with the pie chart
    # size.
    axes_bbox = axs[0,0].get_position()
    width_ax = axes_bbox.width
    height_ax = axes_bbox.height
    radius_inch_max = min(width_ax, height_ax)
    if fig_format == "png":
        plt.savefig(
            f"{root_path}/{name}.{fig_format}",
            dpi=600, transparent=True, bbox_inches=None
        )
    elif fig_format == "pdf":
        plt.savefig(
            f"{root_path}/{name}.{fig_format}"
        )

    # fig 2 legend
    # Create a 2D gradient color map
    x = np.linspace(1, 0, 256)
    y = np.linspace(1, 0, 256)
    X, Y = np.meshgrid(x, y)
    gradient = X
    # height = 80
    mm2inch = 1/25.4 # inch per millimeter
    width = 25
    years = [2030, 2040, 2050]

    fig, axs = plt.subplots(
        row_count, 1,
        figsize=(width*mm2inch, height*mm2inch),
        gridspec_kw={'left': 0.6, 'right': 1.0, 'hspace': 0,
            'top': 0.9, 'bottom': 0.0}
    )

    for te,color in zip(tech, colors):
        y_ix = tech.index(te)
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list(
            'custom',
            [color, "#ffffff"],
            N=10
        )
        color_year = [colormap(6/10),  colormap(3/10), colormap(0/10)]
        
        cmap = ListedColormap(color_year)
        axs[y_ix].imshow(gradient, cmap=cmap, extent=(0, 1, 0, 1),
                        aspect='auto')
        axs[y_ix].set_xticks([])
        axs[y_ix].set_yticks([])
        axs[y_ix].axis("off")
        axs[y_ix].grid(False)

        axs[y_ix].text(x=-0.2,y=0.5,s=te,ha="right",va="center",
                                transform=axs[y_ix].transAxes)

    for p,y in zip([0.333/2+i*0.333 for i in range(3)], years):
        axs[0].text(x=p, y=1.4, s=y,ha="center",va="center",
                        transform=axs[0].transAxes, rotation=90)
    # Show the plot
    plt.subplots_adjust(wspace=0, left=0.08, bottom=0, right=1, top=0.8)  # Set vertical spacing between subplots to 0
    # if fig_format == "png":
    #     plt.savefig(f"{root_path}/%s_legend.%s"%(name, fig_format), dpi=600, transparent=True, bbox_inches=None)
    # elif fig_format == "pdf":
    #     plt.savefig(f"{root_path}/%s_legend.%s"%(name, fig_format))

    # legend 2
    fig, ax = plt.subplots(figsize=(2, 2))
    radius_inch = [map_data_to_size(s, radius_inch_max, max_v)  for s in legend_sizes]
    linewidths = 0.6
    # lw_inch = np.sqrt(linewidths / np.pi) / 72.0
    # print(radius_inch)
    for s,y,ix in zip(legend_sizes, [1, 1-radius_inch[0]+radius_inch[1], 1-radius_inch[0]+radius_inch[2]], [0,1,2]):
        if radius_inch[ix] == float('inf'):
            continue
        circle = plt.Circle((1, y), radius_inch[ix], facecolor='none', edgecolor='black',
                            linewidth=linewidths, alpha=0.5, transform=fig.dpi_scale_trans)
        ax.add_artist(circle)
        yy = y + radius_inch[ix]
        plt.plot([1, 1.2], [yy] * 2, color="black", transform=fig.dpi_scale_trans, lw=linewidths, alpha=0.5)
        ax.text(x=1.2, y=yy, s=f" {s}", transform=fig.dpi_scale_trans, ha="left", va="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    # if fig_format == "png":
    #     plt.savefig(f"{root_path}/{name}_legend2.{fig_format}", dpi=600, transparent=True)
    # elif fig_format == "pdf":
    #     plt.savefig(f"{root_path}/{name}_legend2.{fig_format}")

if __name__ == "__main__":
    font_size = 8
    paper_config = {'axes.labelsize': font_size,
                    'grid.linewidth': 0.2,
                    'font.size': font_size,
                    'legend.fontsize': font_size,
                    'legend.frameon': False,
                    'xtick.labelsize': font_size,
                    'xtick.direction': 'out',
                    'ytick.labelsize': font_size,
                    'ytick.direction': 'out',
                    # 'savefig.bbox': 'tight',
                    'text.usetex': False,
                    'axes.titlesize':font_size,
                    'font.family':'Myriad Pro',
                    'figure.dpi':600,
                    # 'figure.autolayout': True,  # When True, automatically adjust subplot
                                                # parameters to make the plot fit the figure
                                                # using `tight_layout`
                    'savefig.transparent': True}
    plt.rcParams.update(paper_config)
    # dir = "/Users/energy/07-collaboration/03-Xubo/01-xubo-hydro-sediment/revise_stack_bar"
    dir = "."
    generation = pd.read_csv(f"{dir}/output/Elec_balance.csv", index_col=[0,1], header=0)
    main("pdf", generation, "Fig 3B", [2000, 700, 100], dir)