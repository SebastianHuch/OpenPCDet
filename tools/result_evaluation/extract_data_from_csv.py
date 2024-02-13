import os
import csv
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rc
from sklearn.manifold import TSNE
import statsmodels.api as sm
if not sys.version.startswith("3.12"):
    from umap.umap_ import UMAP


fontsize = 20
plt.rcParams["font.size"] = str(fontsize)
rc('axes', linewidth=2)
font_name = "Arial"
plt.rcParams["font.family"] = [font_name]

font = {'family': [font_name],
                'color': 'black',
                'size': fontsize,
                }

def extract_data(runs):
    # Create a directory to store CSV files
    output_dir = f"{network}_output_csv"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize a dict to store max epochs
    epochs = {}

    # Iterate over training datasets and runs
    for run in runs:
        for training_dataset in training_datasets:
            # Iterate over evaluation ranges
            for eval_range in evaluation_ranges:
                # Iterate over evaluation datasets
                for eval_dataset in evaluation_datasets:
                    # Construct the evaluation range folder name
                    eval_range_folder = f"{eval_dataset}_{eval_range}"

                    # Define the path to the evaluation folder for the current evaluation dataset
                    eval_folder = os.path.join(root_dir, network, f"{training_dataset}_{run}", "eval", "eval_all_default", eval_range_folder)

                    # Check if it's a directory
                    if os.path.isdir(eval_folder):
                        # Find the results.csv file
                        results_csv_path = os.path.join(eval_folder, "results.csv")

                        # Check if results.csv exists
                        if os.path.exists(results_csv_path):
                            # Read the CSV file and extract the desired value and epoch
                            with open(results_csv_path, "r") as csv_file:
                                csv_reader = csv.reader(csv_file)

                                # Skip the header line
                                next(csv_reader)

                                values_list = []
                                epochs_list = []

                                for row in csv_reader:
                                    epoch, values, _ = row
                                    epoch = int(epoch)
                                    if iou_thresh == 0.5 and metric == "ap":
                                        value = float(values.split(",")[1])
                                    elif iou_thresh == 0.7 and metric == "ap":
                                        value = float(values.split(",")[3])
                                    elif iou_thresh == 0.5 and metric == "recall":
                                        value = float(values.split(",")[5])*100
                                    elif iou_thresh == 0.7 and metric == "recall":
                                        value = float(values.split(",")[6])*100
                                    else:
                                        raise NotImplementedError(f"IoU Threshold must be 0.5 or 0.7, selected: {iou_thresh}")

                                    # Append the value and epoch to the lists
                                    values_list.append(value)
                                    epochs_list.append(epoch)

                                # Calculate the average, last_value, max_value, and median_value based on last_n_epochs
                                values_list_last_n = values_list[-last_n_epochs:]

                                average_value = sum(values_list_last_n) / len(values_list_last_n)
                                last_value = values_list_last_n[-1]
                                max_value = max(values_list_last_n)
                                median_value = np.median(values_list_last_n)

                                values = {
                                    "avg": average_value,
                                    "max": max_value,
                                    "last": last_value,
                                    "median": median_value
                                }

                                # Store the extracted value along with information about the dataset, eval range folder, and network
                                key = f"{network}_{training_dataset}_{run}_{eval_range_folder}"
                                extracted_data[key] = values[aggregate]
                                extracted_data_list[key] = values_list_last_n  # Store the last_n_epochs values list

                                # Store the epoch separately
                                epochs[key] = epochs_list[-last_n_epochs:][np.asarray(extracted_data_list[key][-last_n_epochs:]).argmax()] if aggregate == "max" else int(epoch)

        # Iterate over evaluation ranges
        for eval_range in evaluation_ranges:
            # Create a CSV file for the current evaluation range (extracted data)
            output_csv_path = os.path.join(output_dir, f"extracted_data_{eval_range}_{run}.csv")

            # Create a CSV file for the current evaluation range (epochs)
            epochs_output_csv_path = os.path.join(output_dir, f"epochs_{eval_range}_{run}.csv")

            # Open the CSV files for writing
            with open(output_csv_path, "w", newline="") as csv_file, open(epochs_output_csv_path, "w",
                                                                          newline="") as epochs_csv_file:
                csv_writer = csv.writer(csv_file)
                epochs_csv_writer = csv.writer(epochs_csv_file)

                # Write the data rows without the first column (training dataset)
                for training_dataset in training_datasets:
                    data_row = []
                    epochs_row = []  # Create a row for epochs

                    for eval_dataset in evaluation_datasets:
                        key = f"{network}_{training_dataset}_{run}_{eval_dataset}_{eval_range}"
                        data_row.append(extracted_data.get(key, ""))
                        epoch = epochs.get(key, "")
                        epochs_row.append(epoch)

                    # Write the rows to the respective CSV files
                    csv_writer.writerow(data_row)
                    epochs_csv_writer.writerow(epochs_row)

    print(f"Extracted data and epoch data saved to {output_dir}")


def create_csv_paper():
    # Calc means and stds from all runs
    means = np.zeros((len(training_datasets), len(evaluation_ranges)))
    stds = np.zeros_like(means)
    for range_idx, eval_range in enumerate(evaluation_ranges):
        # Load data from CSV files for the specified runs
        data_list = []
        for run in runs:
            data = pd.read_csv(f"{network}_output_csv/extracted_data_{eval_range}_{run}.csv", header=None)
            data_list.append(data)

        # Concatenate data from multiple runs
        combined_data = np.asarray(data_list)

        # Calculate mean and standard deviation for each evaluation dataset
        means[:, range_idx] = combined_data.mean(axis=0)[:,0]
        stds[:, range_idx] = combined_data.std(axis=0)[:,0]

    # Create an empty list to store the formatted strings
    formatted_data = []

    # Define a Unicode combining character for the plus symbol above the minus symbol
    plus_minus_symbol = "$\pm$"
    stds_font_size = "\\tiny"

    # Iterate through the means and stds arrays and format the data
    for i in range(means.shape[0]):
        row = []
        for j in range(means.shape[1]):
            mean_str = f"{means[i, j]:.2f}"
            std_str = f"{stds[i, j]:.2f}"
            formatted_value = f"{mean_str}{stds_font_size}{plus_minus_symbol}{std_str}"
            row.append(formatted_value)
        formatted_data.append(row)

    # save to csv
    output_root_dir = f"paper_csv"
    output_dir = os.path.join(output_root_dir, f"{network}")
    os.makedirs(output_dir, exist_ok=True)
    output_csv_paper_path = os.path.join(output_dir, f"{metric}_{str(iou_thresh).replace('.', '')}.csv")

    with open(output_csv_paper_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ')
        csv_writer.writerows(formatted_data)


# Define a function to extract data based on epoch information
def extract_glob_feats(runs):
    # Initialize a dictionary to store extracted data
    extracted_glob_feats = {}

    # Define the output directory for CSVs
    output_dir = f"{network}_output_csv"

    # Iterate over training datasets
    for training_dataset in training_datasets:
        # Iterate over evaluation ranges
        for eval_range in evaluation_ranges:
            # Iterate over evaluation datasets
            for eval_dataset in evaluation_datasets:
                # Construct the evaluation range folder name
                eval_range_folder = f"{eval_dataset}_{eval_range}"

                # Iterate over the specified runs
                for run in runs:
                    # Define the path to the epochs CSV file for the current run
                    epochs_csv_path = os.path.join(output_dir, f"epochs_{eval_range}_{run}.csv")

                    # Check if the epochs CSV file exists
                    if os.path.exists(epochs_csv_path):
                        # Read the epochs CSV file
                        with open(epochs_csv_path, "r") as epochs_csv_file:
                            epochs_csv_reader = csv.reader(epochs_csv_file)

                            for idx, row in enumerate(epochs_csv_reader):
                                if idx == training_datasets.index(training_dataset):
                                    epoch = row[evaluation_datasets.index(eval_dataset)]  # Get the epoch for this dataset
                                    break

                            # Define the folder for the current epoch and run
                            epoch_folder = os.path.join(root_dir, network, f"{training_dataset}_{run}", "eval", "eval_all_default", eval_range_folder, f"epoch_{epoch}")

                            # Check if the epoch folder exists
                            if os.path.exists(epoch_folder):
                                # Define the path to the pkl file
                                if "s2r" in epoch_folder:
                                    pkl_file_path = os.path.join(epoch_folder, "val", f"glob_feat_{epoch}.pkl")
                                else:
                                    pkl_file_path = os.path.join(epoch_folder, "test", f"glob_feat_{epoch}.pkl")

                                # Check if the pkl file exists
                                if os.path.exists(pkl_file_path):
                                    # Load data from the pkl file
                                    with open(pkl_file_path, "rb") as pkl_file:
                                        data = pickle.load(pkl_file)

                                    # remove batch-size dimension
                                    data = list(np.concatenate(data, axis=0))

                                    # Create a key based on dataset, training_dataset, run, and eval_range
                                    key = f"{network}_{training_dataset}_{run}_{eval_range_folder}"

                                    extracted_glob_feats[key] = data

    return extracted_glob_feats


def define_colors():
    # Define the specified colors
    colors = dict()
    for i in range(num_sim):
        colors[i] = tuple(np.asarray(((255/num_sim) * i, 202, 225)) / 255)
    for i in range(num_real):
        colors[num_real - i + num_sim - 1] = tuple(np.asarray((190, (255/num_real) * i / 2, 0)) / 255)
    # sort dict by key
    colors = dict(sorted(colors.items()))
    return colors


def plot_results(colors, runs):
    # Create a directory to store figures
    figs_dir = f"{network}_output_figs"
    os.makedirs(figs_dir, exist_ok=True)

    for eval_range in evaluation_ranges:
        # Load data from CSV files for the specified runs
        data_list = []
        for run in runs:
            data = pd.read_csv(f"{network}_output_csv/extracted_data_{eval_range}_{run}.csv", header=None)
            data_list.append(data)

        # Concatenate data from multiple runs
        combined_data = np.asarray(data_list)

        # Calculate mean and standard deviation for each evaluation dataset
        means = combined_data.mean(axis=0)
        stds = combined_data.std(axis=0)

        # Create a DataFrame for mean and standard deviation data
        mean_df = pd.DataFrame(means)
        std_df = pd.DataFrame(stds)

        # Transpose DataFrames for plotting
        mean_df = mean_df.transpose()
        std_df = std_df.transpose()

        # Set the column names of mean_df and std_df to the training dataset names
        mean_df.columns = training_datasets
        std_df.columns = training_datasets

        # Create a grouped bar plot with adjusted bar width and spacing
        plt.figure(figsize=(24, 12))
        bar_width = 0.2
        num_training_datasets = len(training_datasets)
        index = np.arange(len(evaluation_datasets)) * (num_training_datasets * bar_width + bar_width)

        for i, training_dataset in enumerate(mean_df.columns):
            plt.bar(
                index + i * bar_width,
                mean_df[training_dataset],
                bar_width,
                yerr=std_df[training_dataset],  # Include error bars for standard deviation
                label=f'{training_dataset} (Evaluation)',
                color=colors[i],  # Set the specified bar color
                zorder=2
            )

            # Add the values above each bar
            for j, value in enumerate(mean_df[training_dataset]):
                plt.text(
                    index[j] + i * bar_width,
                    value + 1,  # Adjust the vertical position as needed
                    f"{value:.2f}",  # Format the value with 2 decimal places
                    ha='center',  # Center the text
                    fontsize=8  # Adjust the font size as needed
                )

                # Add the name of each training dataset vertically inside each bar
                plt.text(
                    index[j] + i * bar_width,
                    1,  # Place vertically at the bottom of the bar
                    f'{training_dataset}',  # Training dataset name
                    va='bottom',  # Align the text at the bottom
                    ha='center',  # Center the text
                    fontsize=8,  # Adjust the font size as needed
                    rotation='vertical'  # Rotate the text vertically
                )

        plt.xlabel("Evaluation Dataset")
        plt.ylabel(f"3D AP ({iou_thresh}) in %")
        plt.title(f"3D AP ({iou_thresh}) for Evaluation Range {eval_range} - Aggregate: {aggregate.capitalize()} of last {last_n_epochs} epochs - Runs: {runs}")
        plt.xticks(index + bar_width * (len(mean_df.columns) - 1) / 2, evaluation_datasets, rotation=45)

        # Add a legend that includes both evaluation and training dataset names
        plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0), labels=[f'{training_dataset}' for training_dataset in mean_df.columns], title="Training Dataset")

        # Insert horizontal grid lines at every 10th value on the y-axis
        plt.yticks(np.arange(0, 101, 10))
        plt.grid(axis='y', linestyle='--', which='major', color='gray', linewidth=0.5)

        # Add minor horizontal grid lines without labels every 1 value
        plt.minorticks_on()
        plt.grid(axis='y', linestyle=':', which='minor', color='gray', linewidth=0.5, alpha=0.4)

        # Set y-axis limit from 0 to 100
        plt.ylim(0, 100)

        plt.tight_layout()  # Ensure tight layout to prevent overlapping

        # Save the plot as an image (optional)
        plt.savefig(os.path.join(figs_dir, f"plot_barplot_{eval_range}.png"))

        ################################################################################################################
        ################################################## BOX PLOTS ###################################################
        ################################################################################################################

        # Create a new figure with a single subplot for both boxplots
        fig, axs = plt.subplots(1, len(evaluation_datasets), figsize=(24, 12))

        # Adjust the spacing between subplots
        plt.subplots_adjust(wspace=0.2)  # Adjust the value as needed

        # Iterate over evaluation datasets
        for i, eval_dataset in enumerate(evaluation_datasets):
            # Get the data for the current evaluation dataset from combined_data
            data_by_eval_dataset = combined_data[:, :, i]

            # Create boxplots for the current evaluation dataset on the appropriate subplot
            boxplot = axs[i].boxplot(data_by_eval_dataset, vert=True, patch_artist=True)

            # Set the x-ticks positions and labels
            x_ticks_positions = np.arange(1, len(training_datasets) + 1)
            x_ticks_labels = training_datasets

            axs[i].set_xticks(x_ticks_positions)
            axs[i].set_xticklabels(x_ticks_labels, rotation=30)
            axs[i].tick_params(axis='x', which='both', bottom=True, top=False)
            axs[i].tick_params(axis='x', which='minor', bottom=False)

            # Set the x-axis label
            axs[i].set_xlabel("Training Dataset", labelpad=10)

            axs[i].set_ylabel(f"3D AP ({iou_thresh}) in %")
            axs[i].set_title(f"Evaluation Dataset: {eval_dataset}")
            axs[i].set_ylim(0, 100)

            # Apply the same face colors as the bar plot
            for patch, color in zip(boxplot['boxes'], colors.values()):
                patch.set_facecolor(color)

            # Add horizontal lines at every 10th value on the y-axis
            axs[i].set_yticks(np.arange(0, 101, 10))
            axs[i].grid(axis='y', linestyle='--', which='major', color='gray', linewidth=0.5)

            # Add minor horizontal grid lines without labels every 1 value
            axs[i].minorticks_on()
            axs[i].grid(axis='y', linestyle=':', which='minor', color='gray', linewidth=0.5, alpha=0.4)

            # Calculate and display the median next to each boxplot
            for j, box in enumerate(boxplot['boxes']):
                data = data_by_eval_dataset[:, j]
                median_value = np.median(data)
                axs[i].text(j + 1, median_value + 0.1, f"{median_value:.2f}", ha='center', fontsize=8)

        # Create a custom legend for training datasets outside of the subplots
        legend_patches = [mpatches.Patch(color=color, label=training_dataset) for training_dataset, color in
                          colors.items()]

        # Include the evaluation range in the title of the figure
        fig.suptitle(f"Boxplots for runs {runs} in evaluation range {eval_range} for aggregate '{aggregate}' calculated from last {last_n_epochs} epochs", fontsize=fontsize)

        # Adjust spacing between subplots
        plt.tight_layout()

        # Add the custom legend for training datasets
        fig.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.25, 0.5), title="Training Dataset")

        # Save the plot as an image (optional)
        plt.savefig(os.path.join(figs_dir, f"plot_boxplot_{eval_range}.png"))


def plot_boxplot_paper(colors, runs):
    if num_real + num_sim > 3:
        colors = {0: tuple(np.asarray((158, 202, 225)) / 255),
                  1: tuple(np.asarray((44,162,95)) / 255),
                  2: tuple(np.asarray((44,162,95)) / 255),
                  3: tuple(np.asarray((44,162,95)) / 255),
                  4: tuple(np.asarray((44,162,95)) / 255),
                  5: tuple(np.asarray((49, 130, 189)) / 255),
                  6: tuple(np.asarray((190, 0, 0)) / 255)}
    else:
        colors = {0: tuple(np.asarray((158, 202, 225)) / 255),
                  1: tuple(np.asarray((49, 130, 189)) / 255),
                  2: tuple(np.asarray((255, 158, 132)) / 255),
                  3: tuple(np.asarray((190, 0, 0)) / 255)}

        if num_real > num_sim:
            colors_active = [0, 2, 3]
        else:
            colors_active = [0, 1, 3]

        colors = {num:val for num, val in enumerate([val for key, val in colors.items() if key in colors_active])}

    ranges = ["0.0", "33.3", "66.6", "100.0"]

    # Create a directory to store figures
    figs_dir = f"paper_{network}_figs"
    os.makedirs(figs_dir, exist_ok=True)

    for eval_range in evaluation_ranges:
        # Load data from CSV files for the specified runs
        data_list = []
        for run in runs:
            data = pd.read_csv(f"{network}_output_csv/extracted_data_{eval_range}_{run}.csv", header=None)
            data_list.append(data)

        # Concatenate data from multiple runs
        combined_data = np.asarray(data_list)

        # Iterate over evaluation datasets
        for i, eval_dataset in enumerate(evaluation_datasets):
            # Create a new figure with a single subplot for both boxplots
            if num_real + num_sim > 3:
                figsize = (18, 9)
            else:
                figsize = (7.5, 9)
            fig, axs = plt.subplots(1, 1, figsize=figsize)

            # Font
            plt.rcParams["font.size"] = str(fontsize)
            plt.rcParams["font.family"] = [font_name]

            # Get the data for the current evaluation dataset from combined_data
            data_by_eval_dataset = combined_data[:, :, i]

            # Create beanplot for the current evaluation dataset on the appropriate subplot
            for n in range(len(training_datasets)):
                plot_opts = dict(violin_width=0.8, bean_show_median=False, bean_color='black', jitter_fc='black',
                                 bean_mean_color='black', bean_mean_size=1.0,
                                 violin_fc=colors[n],
                                 violin_alpha=1.0)
                if data_by_eval_dataset.shape[0] == 1:
                    # if only one run selected, duplicate to prevent error in beanplot
                    data_by_eval_dataset = np.tile(data_by_eval_dataset,(2, 1))
                if np.all(data_by_eval_dataset.T[n] == data_by_eval_dataset.T[n][0]):
                    # if values from all 5 runs are identical, add +1e-12 to last run to prevent error in beanplot
                    data_by_eval_dataset.T[n][-1] += 1e-12
                beanplots = sm.graphics.beanplot(data=np.expand_dims(data_by_eval_dataset.T[n], axis=0),
                                                 ax=axs,
                                                 labels=[training_datasets[n]],
                                                 plot_opts=plot_opts,
                                                 positions=[n + 1],
                                                 jitter=True)

            # Horizontal lines for source and target
            source_mean = np.mean(combined_data[:, 0, 0])
            target_mean = np.mean(combined_data[:, -1, 0])
            axs.axhline(y=source_mean, color='k', linestyle='dotted')
            axs.axhline(y=target_mean, color='k', linestyle='dotted')
            axs.text(0.51, source_mean + 1, "Source", ha='left', va='center', color='k', fontsize=12)
            axs.text(0.51, target_mean + 1, "Target", ha='left', va='center', color='k', fontsize=12)

            # Consistent Labels
            labels = []
            if num_real + num_sim > 3:
                for training_dataset in training_datasets:
                    if training_dataset == "real_":
                        labels.append("Real\n(Target)")
                    elif training_dataset == "real2sim20_":
                        labels.append("Real-to-Sim\n$\\delta=3$")
                    elif training_dataset == "real2sim21_":
                        labels.append("Real-to-Sim")
                    elif training_dataset == "real2sim22_":
                        labels.append("Real-to-Sim\nNo-GAN")
                    elif training_dataset == "sim2real22_":
                        labels.append("Sim-to-Real")
                    elif training_dataset == "sim2real21_":
                        labels.append("Sim-to-Real\n$\\delta=$5")
                    elif training_dataset == "sim2real20_":
                        labels.append("Sim-to-Real\n$\\delta=$3")
                    elif training_dataset == "sim2real23_":
                        labels.append("Sim-to-Real\nNo-GAN")
                    elif training_dataset == "sim_noise_obj_" or training_dataset == "sim_noise_":
                        labels.append("Sim-Noise")
                    elif training_dataset == "sim_":
                        labels.append("Sim\n(Source)")
                    else:
                        labels.append("Other")
            else:
                for training_dataset in training_datasets:
                    if training_dataset == "real_":
                        labels.append("Real\n(Target)")
                    elif training_dataset[:8] == "sim2real" and "scene" in training_dataset:
                        labels.append("Sim-to-Real\nScene Full")
                    elif training_dataset[:8] == "sim2real":
                        labels.append("Sim-to-Real\nObject Full")
                    elif training_dataset[:8] == "real2sim":
                        labels.append("Real-to-Sim")
                    elif training_dataset[:9] == "sim_noise":
                        labels.append("Sim-Noise")
                    elif training_dataset == "sim_":
                        labels.append("Sim\n(Source)")

            # Set the x-ticks positions and labels
            x_ticks_positions = np.arange(1, len(training_datasets) + 1)
            x_ticks_labels = labels

            axs.set_xticks(x_ticks_positions)
            axs.set_xticklabels(x_ticks_labels, rotation=00)
            axs.tick_params(axis='x', which='both', bottom=True, top=False)
            axs.tick_params(axis='x', which='minor', bottom=False)
            axs.tick_params(axis='both', which='major', pad=15)

            # Set the x-axis label
            axs.set_xlabel("Training Dataset", labelpad=15, fontdict=font)
            axs.set_xlim(0.45, len(training_datasets) + 0.55)

            range_list = []
            for range_ in eval_range.split("_"):
                for range_lookup in ranges:
                    if int(range_) == int(np.floor(float(range_lookup))):
                        range_list.append(range_lookup)
                        break  # Once a match is found, exit the inner loop

            axs.set_ylabel(f"3D AP ({iou_thresh}) in %", fontdict=font, labelpad=15)
            title = f"Range: [{range_list[0]} m, {range_list[1]} m]" if range_list[1] == "100.0" else f"Range: [{range_list[0]} m, {range_list[1]} m)"
            axs.set_title(title, fontsize=fontsize, pad=15, fontdict=font)
            axs.set_ylim(0, 100)

            # Add horizontal lines at every 10th value on the y-axis
            axs.set_yticks(np.arange(0, 101, 10))
            axs.grid(axis='y', linestyle='--', which='major', color='gray', linewidth=0.5)

            # Add minor horizontal grid lines without labels every 1 value
            axs.minorticks_on()
            axs.grid(axis='y', linestyle=':', which='minor', color='gray', linewidth=0.5, alpha=0.4)

            # Adjust spacing between subplots
            plt.tight_layout()

            # Save the plot as an image (optional)
            if "Sim-Noise" in labels:
                fig_title = f"beanplot_{eval_range}_{eval_dataset}_ablation.pdf"
            else:
                fig_title = f"beanplot_{eval_range}_{eval_dataset}.pdf"
            plt.savefig(os.path.join(figs_dir, fig_title))


def plot_tsne(extracted_glob_feats, evaluation_datasets, training_datasets, evaluation_ranges, runs, colors, perplexity=30):
    # Create a directory to store figures
    figs_dir = f"{network}_output_figs"
    os.makedirs(figs_dir, exist_ok=True)

    # Iterate over evaluation ranges
    for eval_range in evaluation_ranges:
        # Iterate over evaluation datasets
        for eval_dataset in evaluation_datasets:
            # Create a sub-plot for the current evaluation range and dataset
            plt.figure(figsize=(12, 8))
            plt.title(f"T-SNE Plot for Evaluation Range {eval_range} - Dataset {eval_dataset}")

            # Create empty arrays to store t-SNE points and labels
            tsne_points = np.array([])
            tsne_labels = []

            # Iterate over training datasets
            for training_dataset in training_datasets:
                for run in runs:
                    # Define the key for extracted_glob_feats
                    key = f"{network}_{training_dataset}_{run}_{eval_dataset}_{eval_range}"

                    # Check if the key exists in extracted_glob_feats
                    if key in extracted_glob_feats:
                        data = extracted_glob_feats[key]

                        # Convert data to a NumPy array
                        data = np.array(data)

                        # Append the t-SNE points to the array
                        if tsne_points.size == 0:
                            tsne_points = data
                        else:
                            tsne_points = np.vstack((tsne_points, data))

                        # Create labels for the current training dataset
                        labels = np.full(data.shape[0], training_dataset)

                        # Append the labels to tsne_labels
                        tsne_labels.extend(labels)

            # Use t-SNE to reduce the dimensionality to 2
            tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=perplexity, method="barnes_hut", verbose=True, n_iter=500)
            tsne_result = tsne.fit_transform(tsne_points)

            # Scatter plot the t-SNE points with labels
            unique_labels = np.asarray(training_datasets)

            for i, label in enumerate(unique_labels):
                x = tsne_result[np.array(tsne_labels) == label][:, 0]
                y = tsne_result[np.array(tsne_labels) == label][:, 1]
                scat = plt.scatter(x, y, color=colors[i], label=label)
                scat.set_rasterized(True)

                for run_idx, run in enumerate(runs):
                    # Calculate the center of the cluster
                    cluster_center_x = np.median(x[int(run_idx * len(x) / len(runs)) : int((run_idx + 1) * len(x) / len(runs))])
                    cluster_center_y = np.median(y[int(run_idx * len(y) / len(runs)) : int((run_idx + 1) * len(y) / len(runs))])

                    # Add label as text at the cluster center
                    plt.text(cluster_center_x, cluster_center_y, f"{label}_{run}", fontsize=12, ha='center', va='center')

            plt.xlabel("t-SNE Dimension 1")
            plt.ylabel("t-SNE Dimension 2")
            plt.legend()
            plt.grid(True)

            # Save the plot as an image (optional)
            plt.savefig(os.path.join(figs_dir, f"plot_tsne_{eval_dataset}_{eval_range}_{perplexity}.png"))

def plot_tsne_3d(extracted_glob_feats, evaluation_datasets, training_datasets, evaluation_ranges, runs, colors, perplexity=30):
    # Create a directory to store figures
    figs_dir = f"{network}_output_figs"
    os.makedirs(figs_dir, exist_ok=True)

    # Iterate over evaluation ranges
    for eval_range in evaluation_ranges:
        # Iterate over evaluation datasets
        for eval_dataset in evaluation_datasets:
            # Create a sub-plot for the current evaluation range and dataset
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(projection='3d')
            plt.title(f"T-SNE Plot for Evaluation Range {eval_range} - Dataset {eval_dataset}")

            # Create empty arrays to store t-SNE points and labels
            tsne_points = np.array([])
            tsne_labels = []

            # Iterate over training datasets
            for training_dataset in training_datasets:
                for run in runs:
                    # Define the key for extracted_glob_feats
                    key = f"{network}_{training_dataset}_{run}_{eval_dataset}_{eval_range}"

                    # Check if the key exists in extracted_glob_feats
                    if key in extracted_glob_feats:
                        data = extracted_glob_feats[key]

                        # Convert data to a NumPy array
                        data = np.array(data)

                        # Append the t-SNE points to the array
                        if tsne_points.size == 0:
                            tsne_points = data
                        else:
                            tsne_points = np.vstack((tsne_points, data))

                        # Create labels for the current training dataset
                        labels = np.full(data.shape[0], training_dataset)

                        # Append the labels to tsne_labels
                        tsne_labels.extend(labels)

            # Use t-SNE to reduce the dimensionality to 2
            tsne = TSNE(n_components=3, learning_rate='auto', init='pca', perplexity=perplexity, method="barnes_hut", verbose=True, n_iter=500)
            tsne_result = tsne.fit_transform(tsne_points)

            # Scatter plot the t-SNE points with labels
            unique_labels = np.asarray(training_datasets)

            for i, label in enumerate(unique_labels):
                x = tsne_result[np.array(tsne_labels) == label][:, 0]
                y = tsne_result[np.array(tsne_labels) == label][:, 1]
                z = tsne_result[np.array(tsne_labels) == label][:, 2]
                ax.scatter(x, y, z, s=1, color=colors[i], label=label)

                for run_idx, run in enumerate(runs):
                    # Calculate the center of the cluster
                    cluster_center_x = np.median(x[int(run_idx * len(x) / len(runs)) : int((run_idx + 1) * len(x) / len(runs))])
                    cluster_center_y = np.median(y[int(run_idx * len(y) / len(runs)) : int((run_idx + 1) * len(y) / len(runs))])
                    cluster_center_z = np.median(z[int(run_idx * len(z) / len(runs)) : int((run_idx + 1) * len(z) / len(runs))])

                    # Add label as text at the cluster center
                    ax.text(cluster_center_x, cluster_center_y, cluster_center_z + 2, f"{label}_{run}", fontsize=12, ha='center', va='center')

            plt.xlabel("t-SNE Dimension 1")
            plt.ylabel("t-SNE Dimension 2")
            ax.legend()
            ax.grid(True)

            # Save the plot as an image (optional)
            plt.savefig(os.path.join(figs_dir, f"plot_tsne3d_{eval_dataset}_{eval_range}_{perplexity}.png"))


def plot_tsne_paper(extracted_glob_feats, evaluation_datasets, training_datasets, evaluation_ranges, runs, colors, perplexity=30):
    colors = {0: tuple(np.asarray((190, 0, 0)) / 255),
              1: tuple(np.asarray((255, 158, 132)) / 255),
              2: tuple(np.asarray((49, 130, 189)) / 255),
              3: tuple(np.asarray((158, 202, 225)) / 255)}

    if num_real > num_sim:
        colors_active = [0, 1, 3]
    else:
        colors_active = [0, 2, 3]

    colors = {num: val for num, val in enumerate([val for key, val in colors.items() if key in colors_active])}

    # Create a directory to store figures
    figs_dir = f"paper_{network}_figs"
    os.makedirs(figs_dir, exist_ok=True)

    # Iterate over evaluation ranges
    for eval_range in evaluation_ranges:
        # Iterate over evaluation datasets
        for eval_dataset in evaluation_datasets:
            # Create a sub-plot for the current evaluation range and dataset
            fig, ax = plt.subplots(figsize=(12, 12))

            # Font
            plt.rcParams["font.size"] = str(fontsize)
            plt.rcParams["font.family"] = [font_name]

            # Create empty arrays to store t-SNE points and labels
            tsne_points = np.array([])
            tsne_labels = []
            plot_labels = []

            # Iterate over training datasets
            for training_dataset in training_datasets:
                for run in runs:
                    # Define the key for extracted_glob_feats
                    key = f"{network}_{training_dataset}_{run}_{eval_dataset}_{eval_range}"

                    # Check if the key exists in extracted_glob_feats
                    if key in extracted_glob_feats:
                        data = extracted_glob_feats[key]

                        # Convert data to a NumPy array
                        data = np.array(data)

                        # Append the t-SNE points to the array
                        if tsne_points.size == 0:
                            tsne_points = data
                        else:
                            tsne_points = np.vstack((tsne_points, data))

                        # Create labels for the current training dataset
                        labels = np.full(data.shape[0], training_dataset)

                        # Append the labels to tsne_labels
                        tsne_labels.extend(labels)

                    if training_dataset == "real_":
                        plot_labels.append("Real")
                    elif training_dataset[:8] == "sim2real":
                        plot_labels.append("Sim-to-Real")
                    elif training_dataset[:8] == "real2sim":
                        plot_labels.append("Real-to-Sim")
                    elif training_dataset[:9] == "sim_noise":
                        plot_labels.append("Sim-Noise")
                    elif training_dataset == "sim_":
                        plot_labels.append("Sim")

            # Use t-SNE to reduce the dimensionality to 2
            tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=perplexity, method="barnes_hut", verbose=True, n_iter=1000)
            tsne_result = tsne.fit_transform(tsne_points)

            # Scatter plot the t-SNE points with labels
            unique_labels = np.asarray(training_datasets)

            scatter = []
            for i, label in enumerate(unique_labels):
                x = tsne_result[np.array(tsne_labels) == label][:, 0]
                y = tsne_result[np.array(tsne_labels) == label][:, 1]
                scatter.append(ax.scatter(x, y, color=colors[i], label=label))
                scatter[-1].set_rasterized(True)

            plt.legend()
            plt.legend((scatter),
                       plot_labels,
                       scatterpoints=1,
                       markerscale=2,
                       # loc='upper right',
                       ncol=1,
                       fontsize=fontsize)
            plt.grid(True)
            ax.set_axisbelow(True)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.axis('equal')
            plt.subplots_adjust(top=0.99, bottom=0.01, right=0.99, left=0.01, hspace=0, wspace=0)

            # Save the plot as an image (optional)
            plt.savefig(os.path.join(figs_dir, f"plot_tsne_{eval_dataset}_{eval_range}_{perplexity}.pdf"))


def plot_umap_paper(extracted_glob_feats, evaluation_datasets, training_datasets, evaluation_ranges, runs, colors, perplexity=30):
    colors = {0: tuple(np.asarray((190, 0, 0)) / 255),
              1: tuple(np.asarray((255, 158, 132)) / 255),
              2: tuple(np.asarray((49, 130, 189)) / 255),
              3: tuple(np.asarray((158, 202, 225)) / 255)}

    if num_real > num_sim:
        colors_active = [0, 1, 3]
    else:
        colors_active = [0, 2, 3]

    colors = {num: val for num, val in enumerate([val for key, val in colors.items() if key in colors_active])}

    # Create a directory to store figures
    figs_dir = f"paper_{network}_figs"
    os.makedirs(figs_dir, exist_ok=True)

    # Iterate over evaluation ranges
    for eval_range in evaluation_ranges:
        # Iterate over evaluation datasets
        for eval_dataset in evaluation_datasets:
            # Create a sub-plot for the current evaluation range and dataset
            fig, ax = plt.subplots(figsize=(12, 12))

            # Font
            plt.rcParams["font.size"] = str(fontsize)
            plt.rcParams["font.family"] = [font_name]

            # Create empty arrays to store t-SNE points and labels
            umap_points = np.array([])
            tsne_labels = []
            plot_labels = []

            # Iterate over training datasets
            for training_dataset in training_datasets:
                for run in runs:
                    # Define the key for extracted_glob_feats
                    key = f"{network}_{training_dataset}_{run}_{eval_dataset}_{eval_range}"

                    # Check if the key exists in extracted_glob_feats
                    if key in extracted_glob_feats:
                        data = extracted_glob_feats[key]

                        # Convert data to a NumPy array
                        data = np.array(data)

                        # Append the t-SNE points to the array
                        if umap_points.size == 0:
                            umap_points = data
                        else:
                            umap_points = np.vstack((umap_points, data))

                        # Create labels for the current training dataset
                        labels = np.full(data.shape[0], training_dataset)

                        # Append the labels to tsne_labels
                        tsne_labels.extend(labels)

                    if training_dataset == "real_":
                        plot_labels.append("Real")
                    elif training_dataset[:8] == "sim2real":
                        plot_labels.append("Sim-to-Real")
                    elif training_dataset[:8] == "real2sim":
                        plot_labels.append("Real-to-Sim")
                    elif training_dataset[:9] == "sim_noise":
                        plot_labels.append("Sim-Noise")
                    elif training_dataset == "sim_":
                        plot_labels.append("Sim")

            # Use UMAP to reduce the dimensionality to 2
            reducer = UMAP()
            umap_result = reducer.fit_transform(umap_points)

            # Scatter plot the t-SNE points with labels
            unique_labels = np.asarray(training_datasets)

            scatter = []
            for i, label in enumerate(unique_labels):
                x = umap_result[np.array(tsne_labels) == label][:, 0]
                y = umap_result[np.array(tsne_labels) == label][:, 1]
                scatter.append(ax.scatter(x, y, color=colors[i], label=label))
                scatter[-1].set_rasterized(True)

            plt.legend()
            plt.legend((scatter),
                       plot_labels,
                       scatterpoints=1,
                       markerscale=2,
                       # loc='upper right',
                       ncol=1,
                       fontsize=fontsize)
            plt.grid(True)
            ax.set_axisbelow(True)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.axis('equal')
            plt.subplots_adjust(top=0.99, bottom=0.01, right=0.99, left=0.01, hspace=0, wspace=0)

            # Save the plot as an image (optional)
            plt.savefig(os.path.join(figs_dir, f"plot_umap_{eval_dataset}_{eval_range}_{perplexity}.pdf"))


def main():
    colors = define_colors()

    # mAP
    extract_data(runs)
    create_csv_paper()
    #plot_results(colors, runs)
    plot_boxplot_paper(colors, runs)


    # t-SNE
    #extracted_glob_feats = extract_glob_feats(runs)
    #plot_tsne(extracted_glob_feats, evaluation_datasets, training_datasets, evaluation_ranges, runs, colors, perplexity=100)
    #plot_tsne_3d(extracted_glob_feats, evaluation_datasets, training_datasets, evaluation_ranges, runs, colors, perplexity=100)
    #plot_tsne_paper(extracted_glob_feats, evaluation_datasets, training_datasets, evaluation_ranges, runs, colors, perplexity=5)
    #plot_umap_paper(extracted_glob_feats, evaluation_datasets, training_datasets, evaluation_ranges, runs, colors, perplexity=5)

    plt.show()

    print()


if __name__ == "__main__":
    # Dataset association
    # real_             REAL
    # real2sim21_		R2S down5 1485 ONLY R2S
    # sim_noise_obj_	NOISE
    # sim2real20_		S2R down3 1479 sigma3
    # sim2real21_		S2R down5 1483 sigma5
    # sim2real22_		S2R down7 1484 BEST
    # sim2real23_		S2R down7 1503 no-gan
    # sim_			    SIM

    # Define the list of training dataset names
    training_datasets = ["real_", "real2sim21_", "sim_"]
    training_datasets = ["sim_", "sim2real22_", "real_"]
    #training_datasets = ["sim_",
    #                      "sim2real_scene2", "sim2real_scene3", "sim2real_scene1",
    #                     "real_"]
    #training_datasets = ["sim_", "sim2real_scene2", "real_"]
    #training_datasets = ["CARLA", "CARLA2KITTI2", "CARLA2KITTI3", "CARLA2KITTI_SCENE1", "CARLA2KITTI_SCENE2", "CARLA2KITTI_SCENE3", "KITTI"]
    num_real = sum([1 for name in training_datasets if name[:4] == "real" or name[:5] == "KITTI"])
    num_sim = sum([1 for name in training_datasets if name[:3] == "sim" or name[:5] == "CARLA"])
    assert num_sim + num_real == len(training_datasets)

    runs = [1,2,3,4,5]  # [1,2,3,4,5] or [6,7,8,9,10]

    # Define the list of evaluation ranges "0_33", "0_100"
    evaluation_ranges = ["0_33", "0_100"]

    # Define the list of evaluation datasets (real or sim)
    evaluation_datasets = ["real"]

    # Select the network (pointrcnn or pointpillar)
    network = "pointrcnn"

    # Create a dictionary to store the extracted values
    extracted_data = {}
    extracted_data_list = {}

    # Define the root directory
    root_dir = "/Volumes/ge75huw/01_Trainings/01_Sim2Real_OpenPCDet/indy/indy_models"

    # Number of last epochs to consider for averaging/maxing
    last_n_epochs = 10  # Change this as needed

    # avg, max, last, median
    aggregate = "max"

    # ap or recall
    metric = "ap"

    # IoU threshold for 3D AP (0.5 or 0.7)
    iou_thresh = 0.7

    main()
