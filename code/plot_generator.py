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

# Set the incompetent (0) & competent (1) teachers' scores
# Set None to disable
TEACHER_SCORE_0 = 1418.0
TEACHER_SCORE_1 = 8178.0

TAGS = [
    'Evaluation/Reward_Real',
    'Evaluation_B/Reward_Real',
    'Advices_Taken',
    'Advices_Reused/All',
    'Advices_Reused_Correct/All',
    'Advices_Reuse_Model_Correct/Cumulative',
    'Advices_Reuse_Model_Correct/Steps',
]

# Multi-plot settings
# Directory that contain the sub-directories of games with the summary files
RUNS_DIR_MULTI = 'E:\\S\\All\\'
TAGS_MULTI = [
    'Evaluation/Reward_Real',
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
    print('>>> export_to_csv - ', requested_tag)
    if requested_tag != 'Advices_Reuse_Model_Correct/Steps':
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

                if tag == 'Advices_Reuse_Model_Correct-Cumulative':
                    df_values = df.iloc[:, 0].values
                    df_values[1:] -= df_values[:-1].copy()
                    df.iloc[:, 0] = df_values
                    t_resampled = np.linspace(600, 5000000, 2500)
                    df = df.reindex(df.index.union(t_resampled)).interpolate('values').loc[t_resampled]
                    df.to_csv(os.path.join(output_dir, 'Advices_Reuse_Model_Correct-Steps' + '.csv'), header=False)


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

    print('>>> generate_combined_plot - tag:', tag)

    # Plot smoothing span
    if tag == 'Advices_Taken':
        span = 30
    elif tag == 'Advices_Reused/All':
        span = 150
    elif tag == 'Advices_Reuse_Model_Correct/Steps':
        span = 200
    else:
        span = 5

    pda_all, labels = [], []
    run_dirs, plot_dirs = [], []

    for d in next(os.walk(summaries_dir))[1]:
        labels.append(d)
        run_dirs.append(os.path.join(summaries_dir, d))

    # [Algos x 5] - per row: Algo Name, Init, inter, last, and total rewards
    if tag == 'Evaluation/Reward_Real':
        reward_stats = [[j for j in range(5)] for k, _ in enumerate(run_dirs)]
        reward_stats_stddev = [[j for j in range(5)] for k, _ in enumerate(run_dirs)]
        reward_stats_stderr = [[j for j in range(5)] for k, _ in enumerate(run_dirs)]

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
            reward_stats_stddev[i][0] = labels[i]
            reward_stats_stderr[i][0] = labels[i]

            # Average out
            for k in range(4):
                reward_stats[i][k+1] = sum(seed_sum[k]) / len(seed_sum[k])
                reward_stats_stddev[i][k+1] = round(np.std(seed_sum[k]), 2)
                reward_stats_stderr[i][k + 1] = round(stats.sem(seed_sum[k]), 2)

        if len(pds_x) > 0:
            min_length = min(lengths)
            for i in range(len(pds_x)):
                pd_x = pds_x[i][:min_length, ]
                pd_y = pds_y[i][:min_length, ]

                plot_data_x.append(pd_x)
                plot_data_y.append(pd_y)

            pda = pd.DataFrame(plot_data_y, columns=plot_data_x[0])
            pda = pda.ewm(axis=1, span=span).mean()
            pda = pda.melt()

            pda_all.append(pda)
        else:
            pda_all.append(None)

    if not all(v is None for v in pda_all):

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

            with open(os.path.join(plots_dir,'reward_stats_stddev.csv'), 'w', newline='') as f:
                write = csv.writer(f)
                write.writerows(reward_stats_stddev)

            with open(os.path.join(plots_dir,'reward_stats_stderr.csv'), 'w', newline='') as f:
                write = csv.writer(f)
                write.writerows(reward_stats_stderr)

        if tag == 'Evaluation/Reward_Real' or tag == 'Evaluation_B/Reward_Real':
            if TEACHER_SCORE_0 is not None:
                plt.axhline(y=TEACHER_SCORE_0, color='rosybrown', linestyle='--')
            if TEACHER_SCORE_1 is not None:
                plt.axhline(y=TEACHER_SCORE_1, color='darkseagreen', linestyle='--')

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
                if tag == 'Evaluation/Reward_Real' or tag == 'Evaluation_B/Reward_Real':
                    y_lim = [-200, None]
            modify_and_save_plot(ax, tag, x_lim, y_lim, tag.replace("/", "-"))

# ======================================================================================================================

def modify_and_save_plot(ax, tag, x_lim, y_lim, filename):
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    plt.setp(ax.get_legend().get_texts(), fontsize='20')
    plt.setp(ax.get_legend().get_title(), fontsize='25')

    ax_handles, ax_labels = ax.get_legend_handles_labels()

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

def generate_pda(summaries_dir, tags):

    pda_of_tags = {}
    labels_of_tags = {}

    tags += ['Advices_Taken_Long',
             'Advices_Reused/All_Long',
             'Advices_Reused_Correct/All_Long']

    for tag in tags:
        # --------------------------------------------------------------------------------------------------------------
        # Set span here
        span = 20  # Default
        if tag == 'Advices_Taken':
            span = 5
        elif tag == 'Advices_Taken_Cumulative':
            span = 5
        elif tag == 'Evaluation/Reward_Real':
            span = 10
        elif tag == 'Advices_Reused/All' or tag == 'Advices_Reused_Correct/All':
            span = 1000
        elif tag == 'Advices_Reused_Cumulative/All' or tag == 'Advices_Reused_Cumulative_Correct/All':
            span = 1
        elif tag == 'Advices_Taken_Long' or \
                tag == 'Advices_Reused/All_Long' or \
                tag == 'Advices_Reused_Correct/All_Long':
            span = 1000

        # --------------------------------------------------------------------------------------------------------------

        labels = {}
        run_dirs = {}

        for d in next(os.walk(summaries_dir))[1]:
            run_dirs[d] = os.path.join(summaries_dir, d)
            labels[d] = d

        labels_of_tags[tag] = labels

        plot_datas_y = {}
        plot_datas_x = {}
        for key, run_dir in run_dirs.items():

            if tag == 'Advices_Taken_Long' or \
                    tag == 'Advices_Reused_Cumulative/All' or \
                    tag == 'Advices_Reused_Cumulative_Correct/All' or  \
                    tag == 'Advices_Reused/All' or \
                    tag == 'Advices_Reused/All_Long' or \
                    tag == 'Advices_Reused_Correct/All' or \
                    tag == 'Advices_Reused_Correct/All_Long' or \
                    tag == 'Advices_Reused_Percentage' or \
                    tag == 'Advices_Reused_Correct_Percentage':
                if key != 'AIR' and key != 'SUA' and key != 'SUA-AIR' and key != 'DUA':
                    continue
            elif tag == 'Advices_Taken':
                if key != 'RA' and key != 'AIR' and key != 'SUA' and key != 'SUA-AIR' and key != 'DUA':
                    continue

            seed_dirs = []
            for d in next(os.walk(run_dir))[1]:
                seed_dirs.append(os.path.join(run_dir, d))

            plot_data_y = []
            plot_data_x = []

            for seed_dir in seed_dirs:
                if tag == 'Advices_Taken_Long':
                    data_x, data_y = read_data(os.path.join(seed_dir,
                                                            'Advices_Taken'.replace("/", "-") + '.csv'))
                elif tag == 'Advices_Reused/All_Long':
                    data_x, data_y = read_data(os.path.join(seed_dir,
                                                            'Advices_Reused/All'.replace("/", "-") + '.csv'))
                elif tag == 'Advices_Reused_Correct/All_Long':
                    data_x, data_y = read_data(os.path.join(seed_dir,
                                                            'Advices_Reused_Correct/All'.replace("/", "-") + '.csv'))
                else:
                    data_x, data_y = read_data(os.path.join(seed_dir, tag.replace("/", "-") + '.csv'))

                plot_data_y.append(data_y)
                plot_data_x.append(data_x)

            plot_datas_y[key] = plot_data_y
            plot_datas_x[key] = plot_data_x

        pda_all = {}
        for key, plot_data_y in plot_datas_y.items():
            if tag == 'Advices_Taken':
                if key == 'AIR' \
                        or key == 'SUA' \
                        or key == 'SUA-AIR'  \
                        or key == 'DUA' \
                        or key == 'RA':
                    span = 5
                else:
                    span = 1

            pda = pd.DataFrame(plot_data_y, columns=plot_datas_x[key][0])
            pda = pda.ewm(axis=1, span=span).mean()
            pda = pda.melt()
            pda_all[key] = pda

        pda_of_tags[tag] = pda_all

    return pda_of_tags, labels_of_tags

# ======================================================================================================================


def plot_in_multiplot(name, tag, ax, run_idx, labels, pda_all, y_range, x_range, legend, x_label, y_label, title, text,
                      hide_x_ticks, hide_y_ticks):

    print('>>> plot_in_multiplot - name: {}, tag: {}, labels: {}, title: {}'.format(name, tag, labels, title))

    dict_final = {}
    dict_auc = {}

    # Tag dependent
    if tag == 'Evaluation/Reward_Real' \
            or tag == 'Exploration_Steps_Taken' \
            or tag == 'Exploration_Steps_Taken_Cumulative':
        run_idx = ['NA', 'EA', 'AIR', 'SUA', 'SUA-AIR']
    elif tag == 'Advices_Taken':
        run_idx = ['AIR', 'SUA', 'SUA-AIR']
    else:
        run_idx = ['AIR', 'SUA', 'SUA-AIR']

    for run_id in run_idx:
        label = labels[run_id]
        if tag == 'Advices_Reused_Cumulative/All' or tag == 'Advices_Reused/All':
            label = 'All'
        elif tag == 'Advices_Reused_Cumulative_Correct/All' or tag == 'Advices_Reused_Correct/All':
            label = 'Correctly Imitated'
        elif tag == 'Evaluation/Reward_Real':
            if label == 'EA':
                label = 'EA'
            elif label == 'RA':
                label = 'RA'
            elif label == 'None' or label == 'NA':
                label = 'NA'
            elif label == 'AIR':
                label = 'AIR'
            elif label == 'SUA':
                label = 'SUA'
            elif label == 'SUA-AIR':
                label = 'SUA-AIR'
            elif label == 'DUA':
                label = 'DUA'

        # Compute AUC and Final Value
        if tag == 'Evaluation/Reward_Real':
            n_seeds = 3

            # Extract seed value sets
            values = pda_all[run_id]['value'].to_numpy()
            seed_values = []

            seed_aucs = []
            seed_finals = []

            for i_seed in range(n_seeds):
                seed_values.append(values[i_seed::n_seeds])
                seed_aucs.append(np.trapz(values[i_seed::n_seeds])/100)
                seed_finals.append(np.mean(values[i_seed::n_seeds][-3:]))

            # print('{}: {}'.format(labels[run_id], np.mean(seed_aucs)))
            # print('{}: {}'.format(labels[run_id], np.mean(seed_finals)))

            # print('tag1', run_id)

            final_str = '${:.2f} \pm {:.2f}$'.format(round(np.mean(seed_finals), 2), round(np.std(seed_finals), 2))
            auc_str = '${:.2f} \pm {:.2f}$'.format(round(np.mean(seed_aucs), 2), round(np.std(seed_aucs), 2))

            dict_final[run_id] = final_str
            dict_auc[run_id] = auc_str

        if tag == 'Exploration_Steps_Taken_Cumulative' or \
                tag == 'Advices_Reused_Cumulative/All' or  tag == 'Advices_Reused_Cumulative_Percentage' or\
                tag == 'Advices_Reused_Cumulative_Correct/All' or tag == 'Advices_Reused_Correct_Cumulative_Percentage'\
                or tag == 'Advices_Reused/All' or tag == 'Advices_Reused/All':

            n_seeds = 3

            values = pda_all[run_id]['value'].to_numpy()
            seed_final_values = []

            for i_seed in range(n_seeds):
                seed_final_values.append(values[i_seed::n_seeds][-1:])

            final_str = '${:.2f} \pm {:.2f}$'.format(round(np.mean(seed_final_values), 2),
                                                     round(np.std(seed_final_values), 2))

            # print('tag2', run_id)

            dict_final[run_id] = final_str

        if 'None' in label or 'NA' in label:
            color = 'darkslategrey'
            graph = sns.lineplot(x='variable', y='value', data=pda_all[run_id], ax=ax, legend=legend, err_style='band',
                         label=label, color=color, ci='sd')
        else:
            if labels[run_id] == 'EA':
                color = 'tab:brown'
            elif labels[run_id] == 'RA':
                color = 'orchid'
            elif labels[run_id] == 'AIR':
                color = 'tab:green'
            elif labels[run_id] == 'SUA':
                color = 'steelblue'
            elif labels[run_id] == 'SUA-AIR':
                color = 'slateblue'
            elif labels[run_id] == 'DUA':
                color = 'firebrick'

            elif 'All' in label:
                color = 'tab:gray'
            elif 'Correct' in label:
                color = 'tab:green'

            graph = sns.lineplot(x='variable', y='value', data=pda_all[run_id], ax=ax, legend=legend,
                                 err_style='band',
                                 label=label, ci='sd', color=color)  # style="variable", markers=True)

        if label == 'None' or label == 'NA':
            linestyle = '--'
        else:
            linestyle = '-'

        if tag == 'Advices_Taken' or tag == 'Advices_Taken_Long':
            linestyle = '-'

        if tag != 'Tau':
            ax.lines[-1].set_linestyle(linestyle)
            ax.lines[-1].set_linewidth(1.2)

        sns.despine()

    # Save dictionaries
    if dict_final:
        with open(RUNS_DIR_MULTI + '/' + name + '_' + tag.replace("/", "-") + '_FINAL'+ '.txt', 'w') as f:
            for key, value in dict_final.items():
                f.write('%s: %s\n' % (key, value))

    if dict_auc:
        with open(RUNS_DIR_MULTI + '/' + name + '_' + tag.replace("/", "-") + '_AUC' + '.txt', 'w') as f:
            for key, value in dict_auc.items():
                f.write('%s: %s\n' % (key, value))


    # if tag == 'Evaluation/Reward_Real':

    if tag == 'Evaluation/Reward_Real':
        ax.set_xticks([0, 500000, 1000000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000, 4500000, 5000000])

    elif tag == 'Advices_Taken_Long' or \
            tag == 'Advices_Reused/All_Long' or \
            tag == 'Advices_Reused_Correct/All_Long':

        ax.set_xticks([0, 500000, 1000000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000, 4500000, 5000000])

    # Short: 0 -> 500k
    elif tag == 'Advices_Reused/All' or \
            tag == 'Advices_Reused_Correct/All':
        ax.set_xticks([0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000])

    elif tag == 'Advices_Taken':
        ax.set_xticks([0, 25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000])

    # y-ticks
    if tag == 'Evaluation/Reward_Real':
        if 'Asterix' in name:
            ax.set_yticks([0, 5, 10, 15, 20, 25])
        elif 'Breakout' in name:
            ax.set_yticks([0, 20, 40, 60, 80, 100])

        elif 'Freeway' in name:
            ax.set_yticks([0, 5, 10, 15, 20, 25, 30])

        elif 'Pong' in name:
            ax.set_yticks([-20, -15, -10, -5, 0, 5, 10, 15])

        elif 'Seaquest' in name:
            ax.set_yticks([0, 2000, 4000, 6000, 8000, 10000])

    elif tag == 'Advices_Reused_Cumulative/All' or tag == 'Advices_Reused_Cumulative_Correct/All':
        ax.set_yticks([0, 20000, 40000, 60000, 80000, 100000])

    elif tag == 'Advices_Reused_Cumulative/All' or \
            tag == 'Advices_Reused_Cumulative_Correct/All' or \
            tag == 'Advices_Reused/All' or \
            tag == 'Advices_Reused/All_Long' or \
            tag == 'Advices_Reused_Correct/All' or \
            tag == 'Advices_Reused_Percentage' or \
            tag == 'Advices_Reused_Correct_Percentage':
        if 'Seaquest' in name:
            ax.set_yticks([0, 4, 8, 12])

        elif 'Freeway' in name:
            ax.set_yticks([0, 15, 30, 45])

        elif 'Qbert' in name:
            ax.set_yticks([0, 15, 30, 45])

    elif tag == 'Advices_Taken':
        ax.set_yticks([0, 25, 50, 75, 100])

    if legend is not False:
        plt.setp(ax.get_legend().get_texts(), fontsize='20')
        plt.setp(ax.get_legend().get_title(), fontsize='25')
        # ax.legend(loc='right')  # 'upper left'

        ax_handles, ax_labels = ax.get_legend_handles_labels()
        ax_handles_sorted, ax_labels_sorted = [], []

        ax_desired_order = \
            [
            'None',
            'None: No Advising',
            'NA',
            'NA: No Advising',
            'EA',
            'EA: Early Advising',
            'RA',
            'RA: Random Advising',
            'AR',
            'AR: Advice Reuse',
            'AIR',
            'AIR: Advice Imitation & Reuse',
            'SUA',
            'SUA-AIR',
            'DUA',
            'All',
            'Total',
            'Correct',
            'Correctly Imitated'
        ]
        for ax_label_in_order in ax_desired_order:
            if ax_label_in_order in ax_labels:
                ind = ax_labels.index(ax_label_in_order)
                ax_handles_sorted.append(ax_handles[ind])
                ax_labels_sorted.append(ax_labels[ind])

        if tag == 'Evaluation/Reward_Real':
            legend_location = 'lower right'
        else:
            # legend_location = 'upper left'
            legend_location = 'upper right'

        ax.legend(ax_handles_sorted, ax_labels_sorted, loc=legend_location, labelspacing=0.25)  # title='Line'
        # ax.legend(handles[::-1], labels[::-1], loc='right')  # title='Line'

    if x_range is not None:
        ax.set_xlim(x_range)
    if y_range is not None:
        ax.set_ylim(y_range)

    # x-axis tick labels:
    if tag == 'Evaluation/Reward_Real' or \
            tag == 'Advices_Taken_Long' or \
            tag == 'Advices_Reused/All_Long' or \
            tag == 'Advices_Reused_Correct/All_Long':
        xlabels = ['{}'.format(x) + '' for x in graph.get_xticks() / 1000000]
        xlabels[0] = '0'
        xlabels[1] = ''
        xlabels[2] = '1'
        xlabels[3] = ''
        xlabels[4] = '2'
        xlabels[5] = ''
        xlabels[6] = '3'
        xlabels[7] = ''
        xlabels[8] = '4'
        xlabels[9] = ''
        xlabels[10] = '5'
        graph.set_xticklabels(xlabels)

    elif tag == 'Advices_Reused_Cumulative/All' or \
                    tag == 'Advices_Reused_Cumulative_Correct/All' or  \
                    tag == 'Advices_Reused/All' or \
                    tag == 'Advices_Reused_Correct/All' or \
                    tag == 'Advices_Reused_Percentage' or \
                    tag == 'Advices_Reused_Correct_Percentage':
        xlabels = ['{}'.format(x) + '' for x in graph.get_xticks() / 1000000]

        xlabels[0] = '0'
        xlabels[1] = ''
        xlabels[3] = ''
        xlabels[5] = ''
        xlabels[7] = ''
        xlabels[9] = ''
        graph.set_xticklabels(xlabels)

    elif tag == 'Advices_Taken':
        # xlabels = ['{}'.format(int(x)) + '' for x in graph.get_xticks() / 1000]
        xlabels = ['{}'.format(x) + '' for x in graph.get_xticks() / 1000000]
        xlabels[0] = '0'
        xlabels[1] = ''
        xlabels[3] = ''
        xlabels[5] = ''
        xlabels[7] = ''
        #xlabels[0] = '0'
        #xlabels[1] = ''
        #xlabels[3] = ''
        #xlabels[5] = ''
        #xlabels[7] = ''
        #xlabels[9] = ''
        graph.set_xticklabels(xlabels)
    else:
        xlabels = ['{}'.format(x) + '' for x in graph.get_xticks() / 1000]
        xlabels[0] = '0'
        graph.set_xticklabels(xlabels)

    # y-axis tick labels:
    if tag == 'Advices_Reused_Cumulative/All' or tag == 'Advices_Reused_Cumulative_Correct/All':
        ylabels = ['{}k'.format(int(y)) + '' for y in graph.get_yticks() / 1000]
        ylabels[0] = '0'
        graph.set_yticklabels(ylabels)

    if tag == 'Evaluation/Reward_Real':
        if 'Seaquest' in name or 'Qbert' in name:
            ylabels = ['{}k'.format(int(y)) + '' for y in graph.get_yticks() / 1000]
            ylabels[0] = '0'
            graph.set_yticklabels(ylabels)

    if tag == 'Advices_Taken' or \
            tag == 'Advices_Reused_Cumulative/All' or \
            tag == 'Advices_Reused_Cumulative_Correct/All' or \
            tag == 'Advices_Reused/All' or \
            tag == 'Advices_Reused/All_Long' or \
            tag == 'Advices_Reused_Correct/All' or \
            tag == 'Advices_Reused_Percentage' or \
            tag == 'Advices_Reused_Correct_Percentage':
        if 'Seaquest' in name:
            ylabels = ['{}'.format(int(y)) + '' for y in graph.get_yticks()]
            ylabels[0] = '0'
            graph.set_yticklabels(ylabels)

    if x_label is not None:  # 'Millions of steps'
        ax.set(xlabel=x_label)
    else:
        ax.xaxis.label.set_visible(False)

    if y_label is not None:  # 'Millions of steps'
        if tag == 'Tau':
            ax.set(ylabel=y_label)
        else:
            ax.set(ylabel=y_label)
    else:
        ax.yaxis.label.set_visible(False)

    if title is not None:
        ax.set_title(title)

    y_min = ax.get_ylim()[0]
    y_max = ax.get_ylim()[1]
    y_len = y_max - y_min
    x_min = ax.get_xlim()[0]
    x_max = ax.get_xlim()[1]
    x_len = x_max - x_min

    if tag != 'Advices_Reused_Cumulative/All' or  tag != 'Advices_Reused/All':
        ax.grid()

    if hide_x_ticks:
        ax.set_xticklabels([])
    if hide_y_ticks:
        ax.set_yticklabels([])

    plt.tight_layout()

# ======================================================================================================================

def generate_eval_plot(pda_of_tags_all, labels_of_tags_all):

    os.makedirs(plots_dir, exist_ok=True)
    sns.set(font_scale=1.3)

    sns.set_style("white")

    fig = plt.figure(figsize=(15 * 1.5, 2.15 * 1.5), dpi=300)

    gs = fig.add_gridspec(1, 5)
    gs.update(wspace=0.15, hspace=0.2)

    axes = []
    axes.append(fig.add_subplot(gs[0, 0]))
    axes.append(fig.add_subplot(gs[0, 1]))
    axes.append(fig.add_subplot(gs[0, 2]))
    axes.append(fig.add_subplot(gs[0, 3]))
    axes.append(fig.add_subplot(gs[0, 4]))

    # Enduro
    game_id = 0
    tag = 'Evaluation/Reward_Real'
    pda_s = pda_of_tags_all[game_id][tag]
    labels = labels_of_tags_all[game_id][tag]
    plot_in_multiplot('Enduro', tag, axes[game_id], None, labels, pda_s, [-10, 1400], [0, 5000000], False,
                      x_label='Millions of environment steps', y_label='Evaluation score', title='Enduro',
                      text=None, hide_x_ticks=False, hide_y_ticks=False)

    # Freeway
    game_id = 1
    tag = 'Evaluation/Reward_Real'
    pda_s = pda_of_tags_all[game_id][tag]
    labels = labels_of_tags_all[game_id][tag]
    plot_in_multiplot('Freeway', tag, axes[game_id], None, labels, pda_s, [-2, 33], [0, 5000000], 'brief',
                      x_label='Millions of environment steps', y_label=None, title='Freeway',
                      text=None, hide_x_ticks=False, hide_y_ticks=False)

    # Pong
    game_id = 2
    tag = 'Evaluation/Reward_Real'
    pda_s = pda_of_tags_all[game_id][tag]
    labels = labels_of_tags_all[game_id][tag]
    plot_in_multiplot('Pong', tag, axes[game_id], None, labels, pda_s, [-22, 16], [0, 5000000], False,
                      x_label='Millions of environment steps', y_label=None, title='Pong',
                      text=None, hide_x_ticks=False, hide_y_ticks=False)

    # Qbert
    game_id = 3
    tag = 'Evaluation/Reward_Real'
    pda_s = pda_of_tags_all[game_id][tag]
    labels = labels_of_tags_all[game_id][tag]
    plot_in_multiplot('Qbert', tag, axes[game_id], None, labels, pda_s, [-20, 4200], [0, 5000000], False,
                      x_label='Millions of environment steps', y_label=None, title='Q*bert',
                      text=None, hide_x_ticks=False, hide_y_ticks=False)

    # Seaquest
    game_id = 4
    tag = 'Evaluation/Reward_Real'
    pda_s = pda_of_tags_all[game_id][tag]
    labels = labels_of_tags_all[game_id][tag]
    plot_in_multiplot('Seaquest', tag, axes[game_id], None, labels, pda_s, [-100, 10100], [0, 5000000], False,
                      x_label='Millions of environment steps', y_label=None, title='Seaquest',
                      text=None, hide_x_ticks=False, hide_y_ticks=False)

    # fig.suptitle('Evaluation Performance (Scenario I)', size='medium')
    plt.tight_layout()

    fig.savefig(RUNS_DIR_MULTI + '/plots_evaluation' + '.png', bbox_inches='tight')

    fig.clear()
    plt.close(fig)

# ======================================================================================================================

def generate_small_split_budget_plot(pda_of_tags_all, labels_of_tags_all):

    os.makedirs(plots_dir, exist_ok=True)
    sns.set(font_scale=1.2)

    sns.set_style("white")

    fig = plt.figure(figsize=(15 * 1.5, 3 * 1.5), dpi=300)

    gs = fig.add_gridspec(2, 5)
    gs.update(wspace=0.175, hspace=0.25)

    axes = []
    axes.append(fig.add_subplot(gs[0, 0]))
    axes.append(fig.add_subplot(gs[0, 1]))
    axes.append(fig.add_subplot(gs[0, 2]))
    axes.append(fig.add_subplot(gs[0, 3]))
    axes.append(fig.add_subplot(gs[0, 4]))

    axes.append(fig.add_subplot(gs[1, 0]))
    axes.append(fig.add_subplot(gs[1, 1]))
    axes.append(fig.add_subplot(gs[1, 2]))
    axes.append(fig.add_subplot(gs[1, 3]))
    axes.append(fig.add_subplot(gs[1, 4]))

    # ------------------------------------------------------------------------------------------------------------------
    game_id = 0
    game_name = 'Enduro'

    tag = 'Advices_Reused/All_Long'
    pda_s = pda_of_tags_all[game_id][tag]
    labels = labels_of_tags_all[game_id][tag]
    plot_in_multiplot(game_name, tag, axes[0], None, labels, pda_s, [0, 18], [0, 5000000], False,
                      x_label=None, y_label='# of adv. reused', title='Enduro',
                      text=None, hide_x_ticks=False, hide_y_ticks=False)
    axes[0].grid()
    axes[0].grid()

    tag = 'Advices_Taken'
    pda_s = pda_of_tags_all[game_id][tag]
    labels = labels_of_tags_all[game_id][tag]
    plot_in_multiplot(game_name, tag, axes[5], None, labels, pda_s, [0, 110], [0, 200000], False,
                      x_label='Millions of environment steps', y_label='# of adv. taken', title=None,
                      text=None, hide_x_ticks=False, hide_y_ticks=False)
    axes[5].grid()
    axes[5].grid()

    # ------------------------------------------------------------------------------------------------------------------

    game_id = 1
    game_name = 'Freeway'

    tag = 'Advices_Reused/All_Long'
    pda_s = pda_of_tags_all[game_id][tag]
    labels = labels_of_tags_all[game_id][tag]
    plot_in_multiplot(game_name, tag, axes[1], None, labels, pda_s, [0, 52], [0, 5000000], False,
                      x_label=None, y_label=None, title='Freeway',
                      text=None, hide_x_ticks=False, hide_y_ticks=False)
    axes[1].grid()
    axes[1].grid()

    tag = 'Advices_Taken'
    pda_s = pda_of_tags_all[game_id][tag]
    labels = labels_of_tags_all[game_id][tag]
    plot_in_multiplot(game_name, tag, axes[6], None, labels, pda_s, [0, 110], [0, 200000], False,
                      x_label='Millions of environment steps', y_label=None, title=None,
                      text=None, hide_x_ticks=False, hide_y_ticks=False)
    axes[6].grid()
    axes[6].grid()

    # ------------------------------------------------------------------------------------------------------------------

    game_id = 2
    game_name = 'Pong'

    tag = 'Advices_Reused/All_Long'
    pda_s = pda_of_tags_all[game_id][tag]
    labels = labels_of_tags_all[game_id][tag]
    plot_in_multiplot(game_name, tag, axes[2], None, labels, pda_s, [0, 32], [0, 5000000], False,
                      x_label=None, y_label=None, title='Pong',
                      text=None, hide_x_ticks=False, hide_y_ticks=False)
    axes[2].grid()
    axes[2].grid()

    tag = 'Advices_Taken'
    pda_s = pda_of_tags_all[game_id][tag]
    labels = labels_of_tags_all[game_id][tag]
    plot_in_multiplot(game_name, tag, axes[7], None, labels, pda_s, [0, 110], [0, 200000], False,
                      x_label='Millions of environment steps', y_label=None, title=None,
                      text=None, hide_x_ticks=False, hide_y_ticks=False)
    axes[7].grid()
    axes[7].grid()

    # ------------------------------------------------------------------------------------------------------------------

    game_id = 3
    game_name = 'Qbert'

    tag = 'Advices_Reused/All_Long'
    pda_s = pda_of_tags_all[game_id][tag]
    labels = labels_of_tags_all[game_id][tag]
    plot_in_multiplot(game_name, tag, axes[3], None, labels, pda_s, [0, 55], [0, 5000000], False,
                      x_label=None, y_label=None, title='Q*bert',
                      text=None, hide_x_ticks=False, hide_y_ticks=False)
    axes[3].grid()
    axes[3].grid()

    tag = 'Advices_Taken'
    pda_s = pda_of_tags_all[game_id][tag]
    labels = labels_of_tags_all[game_id][tag]
    plot_in_multiplot(game_name, tag, axes[8], None, labels, pda_s, [0, 110], [0, 200000], False,
                      x_label='Millions of environment steps', y_label=None, title=None,
                      text=None, hide_x_ticks=False, hide_y_ticks=False)
    axes[8].grid()
    axes[8].grid()

    # ------------------------------------------------------------------------------------------------------------------

    game_id = 4
    game_name = 'Seaquest'

    tag = 'Advices_Reused/All_Long'
    pda_s = pda_of_tags_all[game_id][tag]
    labels = labels_of_tags_all[game_id][tag]
    plot_in_multiplot(game_name, tag, axes[4], None, labels, pda_s, [0, 14], [0, 5000000], False,
                      x_label=None, y_label=None, title='Seaquest',
                      text=None, hide_x_ticks=False, hide_y_ticks=False)
    axes[4].grid()
    axes[4].grid()

    tag = 'Advices_Taken'
    pda_s = pda_of_tags_all[game_id][tag]
    labels = labels_of_tags_all[game_id][tag]
    plot_in_multiplot(game_name, tag, axes[9], None, labels, pda_s, [0, 110], [0, 200000], 'brief',
                      x_label='Millions of environment steps', y_label=None, title=None,
                      text=None, hide_x_ticks=False, hide_y_ticks=False)
    axes[9].grid()
    axes[9].grid()

    plt.tight_layout()

    fig.savefig(RUNS_DIR_MULTI + '/plots_advice' + '.png', bbox_inches='tight')

    fig.clear()
    plt.close(fig)

# ======================================================================================================================

def generate_multi_plots():
    print('>>> generate_multi_plots...')
    pda_of_tags_all, budget_plots_of_tags_all, labels_of_tags_all = [], [], []

    for game in ['Enduro', 'Freeway', 'Pong', 'Qbert', 'Seaquest']:
        print(game)
        summaries_dir = os.path.join(RUNS_DIR_MULTI, game)
        pda_of_tags, labels_of_tags = generate_pda(summaries_dir, TAGS_MULTI)
        pda_of_tags_all.append(pda_of_tags)
        labels_of_tags_all.append(labels_of_tags)

    generate_eval_plot(pda_of_tags_all, labels_of_tags_all)
    generate_small_split_budget_plot(pda_of_tags_all, labels_of_tags_all)

# ======================================================================================================================

if RUNS_DIR != None or GAME_DIR != None:
    summaries_dir = os.path.join(RUNS_DIR, GAME_DIR)
else:
    summaries_dir = SUMM_DIR

if os.path.isdir(summaries_dir) and len(os.listdir(summaries_dir)) != 0:
    plots_dir = os.path.join(summaries_dir + '_Plots')

    if os.path.exists(plots_dir):
        shutil.rmtree(plots_dir)

    #generate_csv_files(summaries_dir, TAGS)

    generate_multi_plots()

    for tag in TAGS:
        generate_combined_plot(summaries_dir, plots_dir, tag)

