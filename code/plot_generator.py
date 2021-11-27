import os
import numpy as np
import pandas as pd
import shutil
import csv

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

# Either provide runs & games directory
RUNS_DIR = None  # 'E:\\Runs'
GAME_DIR = None  # 'Seaquest'

SUMM_DIR = 'D:/UoA/After_Omen_BreakDown_2021/Results/Twin_DQN_3300_5100_6100/Summaries'
# OR provide a direct summaries directory

# Set the environment to plot results for
ENV = 'Seaquest' # 'Seaquest'

#Set the teacher's score for that
TEACHER_SCORE = 12.0  #8178.0

TAGS = [
    'Evaluation/Reward_Real',
    'Evaluation_B/Reward_Real',
    'Advices_Taken',
    'Advices_Reused/All'
]

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

def export_to_csv(input_dir, output_dir, requested_tag):
    print(input_dir, output_dir, requested_tag)
    summary_iterator = EventAccumulator(input_dir).Reload()
    tags = summary_iterator.Tags()['scalars']
    data = defaultdict(list)

    if requested_tag in tags:
        steps = [e.step for e in summary_iterator.Scalars(requested_tag)]
        event = summary_iterator.Scalars(requested_tag)
        data[requested_tag] = [e.value for e in event]

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
                    export_to_csv(seed_dir, seed_dir, tag[0])

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

    # Plot smoothing span
    if tag == 'Advices_Taken':
        span = 30
    elif tag == 'Advices_Reused/All':
        span = 150
    else:
        span = 5

    pda_all, labels = [], []
    run_dirs, plot_dirs = [], []

    for d in next(os.walk(summaries_dir))[1]:
        labels.append(d)
        run_dirs.append(os.path.join(summaries_dir, d))

    # [Algos x 5] - per row: Algo Name, Init, inter, last, and total rewards
    reward_stats = [[j for j in range(5)] for k, _ in enumerate(run_dirs)]

    for i, run_dir in enumerate(run_dirs):
        seed_dirs = []
        for d in next(os.walk(run_dir))[1]:
            seed_dirs.append(os.path.join(run_dir, d))

        plot_data_x, plot_data_y = [], []
        pds_x, pds_y = [], []
        lengths = []

        if tag == 'Evaluation/Reward_Real':
            # [5 x Seed] - each column would be one seed (easy to average across seeds/columns)
            seed_sum = [[j for j in range(len(seed_dirs))] for k in range(4)]

        j = 0
        for seed_dir in seed_dirs:
            csv_filepath = os.path.join(seed_dir, tag.replace("/", "-") + '.csv')
            if os.path.isfile(csv_filepath):
                pd_x, pd_y = read_data(csv_filepath)
                pds_x.append(pd_x)
                pds_y.append(pd_y)
                lengths.append(np.shape(pd_x)[0])

                if tag == 'Evaluation/Reward_Real':
                    l = len(pd_y)

                    # Take a sum for each seed and then average outside x 4
                    seed_sum[0][j] = sum(pd_y[1 : int(l / 3)])
                    seed_sum[1][j] = sum(pd_y[int(l / 3):int(2/3 * l)])
                    seed_sum[2][j] = sum(pd_y[int(2 / 3 * l):l])
                    seed_sum[3][j] = sum(pd_y[1:])

                    j += 1

        if tag == 'Evaluation/Reward_Real':
            # Set the Algo name
            reward_stats[i][0] = labels[i] 
            # Average out
            for k in range(4):
                reward_stats[i][k+1] = sum(seed_sum[k]) / len(seed_sum[k])


        if len(pds_x) > 0:
            min_length = min(lengths)
            for i in range(len(pds_x)):
                pd_x = pds_x[i][:min_length, ]
                pd_y = pds_y[i][:min_length, ]

                plot_data_x.append(pd_x)
                plot_data_y.append(pd_y)

            print(run_dir, tag)

            pda = pd.DataFrame(plot_data_y, columns=plot_data_x[0])
            pda = pda.ewm(axis=1, span=span).mean()
            pda = pda.melt()

            pda_all.append(pda)
        else:
            pda_all.append(None)

    if tag == 'Advices_Taken':
        fig, ax = plt.subplots(figsize=(12, 4), dpi=150, )
    else:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100, )

    for i, _ in enumerate(pda_all):
        if pda_all[i] is not None:
            sns.lineplot(x='variable', y='value', data=pda_all[i], legend='brief', err_style='band',
                         label=labels[i], ci='sd', linewidth=1)

    if tag == 'Evaluation/Reward_Real':
        with open(os.path.join(plots_dir,'reward_stats.csv'), 'w', newline='') as f: 
            write = csv.writer(f) 
            write.writerows(reward_stats)

    if tag == 'Evaluation/Reward_Real' or tag == 'Evaluation_B/Reward_Real':
        plt.axhline(y=TEACHER_SCORE, color='slategray', linestyle='--')

    x_lim, y_lim = [None, None], [None, None]

    if tag == 'Advices_Taken' or \
            tag == 'Advices_Reused/All':
        y_lim = [-5, None]
        x_lim = [-10, None]

        modify_and_save_plot(ax, tag, x_lim, y_lim, tag.replace("/", "-"))

        if tag == 'Advices_Taken':
            x_lim = [-10, 250000]
            modify_and_save_plot(ax, tag, x_lim, y_lim, tag.replace("/", "-") + '_Zoomed')
    else:
        if ENV == 'Seaquest':
            y_lim = [-200, None]
        modify_and_save_plot(ax, tag, x_lim, y_lim, tag.replace("/", "-"))

# ======================================================================================================================

def modify_and_save_plot(ax, tag, x_lim, y_lim, filename):
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    plt.setp(ax.get_legend().get_texts(), fontsize='20')
    plt.setp(ax.get_legend().get_title(), fontsize='25')

    ax_handles, ax_labels = ax.get_legend_handles_labels()

    print(ax_handles, ax_labels)

    legend_loc = 'upper left'
    if tag == 'Advices_Taken':
        legend_loc = 'upper right'
    ax.legend(ax_handles, ax_labels, loc=legend_loc, labelspacing=0.25)

    plt.tight_layout()
    fig = ax.get_figure()

    ax.xaxis.label.set_visible(False)
    ax.yaxis.label.set_visible(False)

    fig.savefig(os.path.join(plots_dir, filename + '.png'))

    #fig.clear()
    #plt.close(fig)

# ======================================================================================================================

if RUNS_DIR != None or GAME_DIR != None:
    summaries_dir = os.path.join(RUNS_DIR, GAME_DIR)
else:
    summaries_dir = SUMM_DIR

if os.path.isdir(summaries_dir) and len(os.listdir(summaries_dir)) != 0:
    plots_dir = os.path.join(summaries_dir + '_Plots')

    if os.path.exists(plots_dir):
        shutil.rmtree(plots_dir)

    generate_csv_files(summaries_dir, TAGS)

    for tag in TAGS:
        generate_combined_plot(summaries_dir, plots_dir, tag)
