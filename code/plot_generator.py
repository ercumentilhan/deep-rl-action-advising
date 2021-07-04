import os
import numpy as np
import pandas as pd
import shutil

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib.pyplot as plt
import seaborn
import scipy
from scipy import stats
import matplotlib.ticker as ticker

import seaborn as sns
sns.set()
sns.set_style("darkgrid", {"axes.facecolor": ".9"})


# ======================================================================================================================

def read_data(csv_file):
    with open(csv_file) as f:
        lines = (line for line in f if not line.startswith('#'))
        data = np.loadtxt(lines, delimiter=',')
        if np.shape(data)[0] == 0:
            return None, None
        return data[:, 0], data[:, 1]

# ======================================================================================================================

def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]

# ======================================================================================================================

def export_to_csv(input_dir, output_dir, requested_tags):
    print(input_dir, output_dir, requested_tags)
    summary_iterator = EventAccumulator(input_dir).Reload()
    tags = summary_iterator.Tags()['scalars']
    data = defaultdict(list)
    steps = []
    for tag in tags:
        if tag in requested_tags:
            steps = [e.step for e in summary_iterator.Scalars(tag)]
            event = summary_iterator.Scalars(tag)
            data[tag] = [e.value for e in event]

    tags, values = zip(*data.items())
    np_values = np.array(values)
    for index, tag in enumerate(tags):
        tag = tag.replace("/", "-")
        df = pd.DataFrame(np_values[index], index=steps)
        df.to_csv(os.path.join(output_dir, tag + '.csv'), header=False)

# ======================================================================================================================

def remove_csv_files(summaries_dir):
    run_dirs = [summaries_dir + '/' + d for d in next(os.walk(summaries_dir))[1]]
    for run_dir in run_dirs:
        seed_dirs = [run_dir + '/' + d for d in next(os.walk(run_dir))[1]]
        for seed_dir in seed_dirs:
            items = os.listdir(seed_dir)
            for item in items:
                if item.endswith('.csv'):
                    os.remove(os.path.join(seed_dir, item))

# ======================================================================================================================

def generate_csv_files(summaries_dir, requested_tags):
    run_dirs = [summaries_dir + '/' + d for d in next(os.walk(summaries_dir))[1]]
    for run_dir in run_dirs:
        seed_dirs = [run_dir + '/' + d for d in next(os.walk(run_dir))[1]]
        for seed_dir in seed_dirs:
            for requested_tag in requested_tags:
                tag = [requested_tag,]
                if not os.path.exists(seed_dir + '/' + tag[0].replace("/", "-") + '.csv'):
                    export_to_csv(seed_dir, seed_dir, tag)

# ======================================================================================================================

def generate_plots(summaries_dir, plots_dir, tag):
    os.makedirs(plots_dir, exist_ok=True)

    span = 5  # Plot smoothing span

    run_dirs, plot_dirs, labels = [], [], []
    for d in next(os.walk(summaries_dir))[1]:
        run_dirs.append(os.path.join(summaries_dir, d))
        plot_dirs.append(os.path.join(plots_dir, d))
        os.makedirs(plot_dirs[-1], exist_ok=True)
        labels.append(d)

    for i, run_dir in enumerate(run_dirs):
        seed_dirs = []
        for d in next(os.walk(run_dir))[1]:
            seed_dirs.append(os.path.join(run_dir, d))

        plot_data_x, plot_data_y = [], []

        for seed_dir in seed_dirs:
            pd_x, pd_y = read_data(os.path.join(seed_dir, tag.replace("/", "-") + '.csv'))
            plot_data_x.append(pd_x)
            plot_data_y.append(pd_y)

        pda = pd.DataFrame(plot_data_y, columns=plot_data_x[0])
        pda = pda.ewm(axis=1, span=span).mean()
        pda = pda.melt()

        fig, ax = plt.subplots(figsize=(12, 8), dpi=100,)

        sns.lineplot(x='variable', y='value', data=pda, legend='brief', err_style='band', label=labels[i], ci='sd')

        # Adjust legend
        plt.setp(ax.get_legend().get_texts(), fontsize='20')
        plt.setp(ax.get_legend().get_title(), fontsize='25')

        ax_handles, ax_labels = ax.get_legend_handles_labels()
        ax.legend(ax_handles, ax_labels, loc='lower right', labelspacing=0.25)

        plt.tight_layout()
        fig = ax.get_figure()

        ax.xaxis.label.set_visible(False)
        ax.yaxis.label.set_visible(False)

        fig.savefig(os.path.join(plot_dirs[i], tag.replace("/", "-") + '.png'))

        fig.clear()
        plt.close(fig)

# ======================================================================================================================

def generate_combined_plot(summaries_dir, plots_dir, tag):
    os.makedirs(plots_dir, exist_ok=True)

    span = 5  # Plot smoothing span

    pda_all, labels = [], []
    run_dirs, plot_dirs = [], []
    for d in next(os.walk(summaries_dir))[1]:
        labels.append(d)
        run_dirs.append(os.path.join(summaries_dir, d))

    for i, run_dir in enumerate(run_dirs):
        seed_dirs = []
        for d in next(os.walk(run_dir))[1]:
            seed_dirs.append(os.path.join(run_dir, d))

        plot_data_x, plot_data_y = [], []

        for seed_dir in seed_dirs:
            pd_x, pd_y = read_data(os.path.join(seed_dir, tag.replace("/", "-") + '.csv'))
            plot_data_x.append(pd_x)
            plot_data_y.append(pd_y)

        pda = pd.DataFrame(plot_data_y, columns=plot_data_x[0])
        pda = pda.ewm(axis=1, span=span).mean()
        pda = pda.melt()

        pda_all.append(pda)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=100, )

    for i, _ in enumerate(run_dirs):
        sns.lineplot(x='variable', y='value', data=pda_all[i], legend='brief', err_style='band',
                     label=labels[i], ci='sd')

    ax.set_ylim([-200, None])

    # Adjust legend
    plt.setp(ax.get_legend().get_texts(), fontsize='20')
    plt.setp(ax.get_legend().get_title(), fontsize='25')

    ax_handles, ax_labels = ax.get_legend_handles_labels()
    ax.legend(ax_handles, ax_labels, loc='lower right', labelspacing=0.25)

    plt.tight_layout()
    fig = ax.get_figure()

    ax.xaxis.label.set_visible(False)
    ax.yaxis.label.set_visible(False)

    fig.savefig(os.path.join(plots_dir, tag.replace("/", "-") + '.png'))

    fig.clear()
    plt.close(fig)

# ======================================================================================================================


runs_dir = 'E:/Runs/Summaries/LunarLander'
game_dir = 'B25000'
summaries_dir = os.path.join(runs_dir, game_dir)

if os.path.isdir(summaries_dir) and len(os.listdir(summaries_dir)) != 0:
    plots_dir = os.path.join(runs_dir, game_dir + '_Plots')
    tags = [
        'Reward_Real/Steps',
        'Evaluation/Reward_Real',
        'Advices_Taken'
            ]

    if os.path.exists(plots_dir):
        shutil.rmtree(plots_dir)

    generate_csv_files(summaries_dir, tags)

    for tag in tags:
        generate_plots(summaries_dir, plots_dir, tag)
        generate_combined_plot(summaries_dir, plots_dir, tag)
