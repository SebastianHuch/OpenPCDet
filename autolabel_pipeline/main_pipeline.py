
# Import libraries
from easydict import EasyDict
import subprocess
import pathlib
import shutil
import yaml
import csv
import os

from base_functions import load_config, autolabel_path_manager
from main_autolabel import main_pseudo_label
from evaluate_labels import get_label_anno, main_evaluate_labels
from pcdet.datasets.autolabel.autolabel_dataset import create_autolabel_infos

# Define a working path used to access different paths
working_path = pathlib.Path(__file__).resolve().parents[1]



"""

FILE DESCRIPTION: 

This file is the main file for the semi-supervised learning pipeline. It outputs labeled data. 
It serves as a central script, triggering respective sub-functions of the pipeline.
--> Initial model training, file management, object prediction, voting, train-set update. 

All parameters are configurable in XX 

voting schemes: 
    - non maximum suppression + confidence
    - majority voting

"""

# See later: Function new project.
#def new_project():
    # check if a new project is intended to be started.
#    if not os.path.exists(os.path.join(working_path,cfg.DATA.PROJECT.AUTOLABEL_DATA)):
 #       manual_continue = input("Do you intend to create a new project? "
#                                "(A new directory will be created at cfg.DATA.PROJECT.AUTOLABEL_DATA. Please check.) y/ n ? ")
#        if manual_continue != "y":
#            exit()
#        else:
#            os.mkdir(os.path.join(working_path,cfg.DATA.PROJECT.AUTOLABEL_DATA))
#    return

# Function that moves the model files from models (autolabel_data) to past_iterations folder.
def prepare_autolabel_data_models_folder(cfg_autolabel, path_manager):
    # Function creates past_iteration path. Subsequently, creates folders to save the used models and iteration XX.
    def iteration_folder(path_past_iterations):

        if not os.path.exists(path_past_iterations):
            os.mkdir(path_past_iterations)
            os.mkdir(os.path.join(path_past_iterations, "iteration_00"))
            path_current_iteration = os.path.join(path_past_iterations, "iteration_00")
        else:
            subdirectories = [item for item in os.listdir(path_past_iterations) if
                              os.path.isdir(os.path.join(path_past_iterations, item))]
            if not subdirectories:
                os.mkdir(os.path.join(path_past_iterations, "iteration_00"))
                path_current_iteration = os.path.join(path_past_iterations, "iteration_00")
            else:
                iteration_numbers = [int(subdir.split('_')[1]) for subdir in subdirectories if
                                     subdir.startswith("iteration_")]
                if iteration_numbers:
                    next_iteration_number = max(iteration_numbers) + 1
                else:
                    next_iteration_number = 0
                next_iteration_folder = f"iteration_{next_iteration_number:02}"
                os.mkdir(os.path.join(path_past_iterations, next_iteration_folder))
                path_current_iteration = os.path.join(path_past_iterations, next_iteration_folder)
        return path_current_iteration
    # Function empties a folder.
    def empty_folder(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
        return

    print("-> prepare autolabel data folder")
    path_past_iterations = os.path.join(path_manager.get_path("path_model_ckpt_dir"), "past_iterations")
    path_current_iteration = iteration_folder(path_past_iterations)
    print(f"Save models of past iteration in {path_current_iteration}")


    # Copy folders to iteration_XX folder
    shutil.copytree(os.path.join(path_manager.get_path("path_model_ckpt_dir"), "pointrcnn"),
                    os.path.join(path_current_iteration, "pointrcnn"))
    shutil.copytree(os.path.join(path_manager.get_path("path_model_ckpt_dir"), "pointpillar"),
                    os.path.join(path_current_iteration, "pointpillar"))
    shutil.copytree(os.path.join(path_manager.get_path("path_model_ckpt_dir"), "second"),
                    os.path.join(path_current_iteration, "second"))

    # Remove the contents of models/
    empty_folder(os.path.join(path_manager.get_path("path_model_ckpt_dir"), "pointrcnn"))
    empty_folder(os.path.join(path_manager.get_path("path_model_ckpt_dir"), "pointpillar"))
    empty_folder(os.path.join(path_manager.get_path("path_model_ckpt_dir"), "second"))

    # Replace models/ ckpts with new selected ckpts.
    selected_ckpts = []
    for model in cfg_autolabel.DATA.PROJECT.MODELS:
        this_ckpt = input(f"Input checkpoint epoch to load for model {model} (checkpoint_  .pth is added automatically): ")
        selected_ckpts.append("checkpoint_epoch_" + this_ckpt + ".pth")
    print(f"Selected model checkpoints: {selected_ckpts} for models {cfg_autolabel.DATA.PROJECT.MODELS}")

    confirmation = input(f"Confirm your selection (y/n): ")
    if confirmation == "y":
        for element in range(len(cfg_autolabel.DATA.PROJECT.MODELS)):
            ckpt = selected_ckpts[element]
            model = cfg_autolabel.DATA.PROJECT.MODELS[element]

            if model == "pointrcnn":
                path_ckpt_source = os.path.join(path_manager.get_path("path_pcdet_pointrcnn"), ckpt)
                path_ckpt_destination = os.path.join(path_manager.get_path("path_model_ckpt_dir"), "pointrcnn")

            elif model == "pointpillar":
                path_ckpt_source = os.path.join(path_manager.get_path("path_pcdet_pointpillar"), ckpt)
                path_ckpt_destination = os.path.join(path_manager.get_path("path_model_ckpt_dir"), "pointpillar")

            elif model == "second":
                path_ckpt_source = os.path.join(path_manager.get_path("path_pcdet_second"), ckpt)
                path_ckpt_destination = os.path.join(path_manager.get_path("path_model_ckpt_dir"), "second")

            else:
                raise ValueError("Error in prepare_autolabel_data_folder. ")

            shutil.copy(path_ckpt_source, path_ckpt_destination)

    else:
        exit()
    return

def reset_mode2_folders(cfg_autolabel, path_manager, models):
    print("-> Resetting folders utilized within mode2.")

    # FOR PREDICTIONS: Remove existing prediction folder to have only predictions specified.
    for model in models:
        if model == "pointrcnn":
            path_predictions = path_manager.get_path("path_ptrcnn_predictions")
        elif model == "pointpillar":
            path_predictions = path_manager.get_path("path_ptpillar_predictions")
        elif model == "second":
            path_predictions = path_manager.get_path("path_second_predictions")
        else:
            raise ValuError("Model not existing. Check cfg_autolabel.DATA.PROJECT.MODELS.")

        if os.path.exists(path_predictions):
            try:
                shutil.rmtree(path_predictions)
            except Exception as e:
                print(f"An error occurred while trying to remove old contents of path_predictions: {e}")

    # FOR VOTE_PSEUDO_LABELS:
    if cfg_autolabel.PIPELINE.VOTING_SCHEME == "MAJORITY":
        path_pseudo_labels = path_manager.get_path("path_pseudo_labels_majority")
    elif cfg_autolabel.PIPELINE.VOTING_SCHEME == "NMS":
        path_pseudo_labels = path_manager.get_path("path_pseudo_labels_nms")
    else:
        raise ValueError("Voting scheme not existing. Check cfg_autolabel.PIPELINE.VOTING_SCHEME.")

    if os.path.exists(path_pseudo_labels):
        try:
            shutil.rmtree(path_pseudo_labels)
        except Exception as e:
            print(f"An error occurred while trying to remove old contents of path_pseudo_labels: {e}")

    # FOR EVALUATION_METRICS:
    path_evaluation = os.path.join(path_manager.get_path("path_project_data"), "evaluation")
    if os.path.exists(path_evaluation):
        try:
            shutil.rmtree(path_evaluation)
        except Exception as e:
            print(f"An error occurred while trying to remove old contents of path_evaluation: {e}")

    # FOR CONVERTED OUTPUT LABELS:
    path_output_labels = path_manager.get_path("path_output_labels")
    if os.path.exists(path_output_labels):
        try:
            shutil.rmtree(path_output_labels)
        except Exception as e:
            print(f"An error occurred while trying to remove old contents of path_output_labels: {e}")

    return



# MODE 1 sub-functions
# Function that updates the config file of the detector model. (NUM_EPOCHS; EVAL_RATE; EVAL_MIN_EPOCH).
def cfg_train_model_update(path_manager, model, print_cfg):

    cfg_file_model_path = os.path.join(path_manager.get_path("path_cfg_models"), (model + ".yaml"))

    with open(cfg_file_model_path, 'r') as cfg_file:
        cfg_data = yaml.safe_load(cfg_file)
    # Rate at which evaluation is triggered. Every X epochs.
    cfg_data["OPTIMIZATION"]["EVAL_RATE"] = 4
    # Minimum number of training epochs before evaluating.
    cfg_data["OPTIMIZATION"]["EVAL_MIN_EPOCH"] = 54

    # Model specific parameters, adapted to autolabel needs.
    if model == "pointrcnn":
        cfg_data["OPTIMIZATION"]["NUM_EPOCHS"] = 75 #85
        cfg_data['OPTIMIZATION']['BATCH_SIZE_PER_GPU'] = 6

    if model == "pointpillar":
        cfg_data["OPTIMIZATION"]["NUM_EPOCHS"] = 75 #85
        cfg_data['OPTIMIZATION']['BATCH_SIZE_PER_GPU'] = 10

    if model == "second":
        cfg_data["OPTIMIZATION"]["NUM_EPOCHS"] = 80 #90
        cfg_data['OPTIMIZATION']['BATCH_SIZE_PER_GPU'] = 12

    with open(cfg_file_model_path, 'w') as updated_cfg_file:
        yaml.dump(cfg_data, updated_cfg_file)

    if print_cfg:
        with open(cfg_file_model_path, 'r') as cfg_file:
            cfg_data = yaml.safe_load(cfg_file)
            print(yaml.dump(cfg_data))

# Function that triggers openPCDet training.
def model_training(working_path, path_manager, model):
    print("-> Training.")

    cfg_file_model_path = os.path.join(path_manager.get_path("path_cfg_models"), (model + ".yaml"))

    train_path = os.path.join(working_path, "tools")
    train_script_path = os.path.join(train_path, "train.py")
    os.chdir(train_path)
    print("Working directory for training: ", os.getcwd())

    command = f"python {train_script_path} --cfg_file {cfg_file_model_path}"
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while running the training process: {e}")

# Function that copies a model ckpt to a specified folder.
def copy_pth_file(model_file, origin_path, goal_path):
    source_file_path = os.path.join(origin_path, model_file)
    destination_file_path = os.path.join(goal_path, model_file)

    if os.path.isfile(source_file_path):
        if os.path.exists(destination_file_path) and len(os.listdir(goal_path)) > 0:
            user_input = input(f"Destination folder {goal_path} is not empty. Do you want to empty it? (y/n): ").lower()
            if user_input == 'y':
                try:
                    for file in os.listdir(goal_path):
                        file_path = os.path.join(goal_path, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    print(f"Destination folder {goal_path} emptied.")
                except Exception as e:
                    print(f"Error occurred while emptying the destination folder: {e}")
            else:
                print("Aborted. Destination folder is not empty.")
                return

        try:
            shutil.copy(source_file_path, destination_file_path)
            print(f"Successfully copied {model_file} from {origin_path} to {goal_path}.")
        except Exception as e:
            print(f"Error occurred while copying {model_file}: {e}")
    else:
        print(f"Source file {model_file} does not exist in {origin_path}.")



# MODE 2 sub-functions
# Function that counts the total number of files in a folder.
def count_file_number_in_folder(this_folder):

    csv_count = 0
    file_list = os.listdir(this_folder)
    for filename in file_list:
        if filename.endswith(".txt"):
            csv_count += 1

    print(f"Number of pseudo-labels usable for training in '{this_folder}': {csv_count}")
    return

# Function that updates the config file of the dataset used.
def cfg_dataset_update(cfg_autolabel, path_manager, print_cfg, eval_on_val):

    if eval_on_val:
        print("-> Updating dataset cfg for eval on val.")
    else:
        print("-> Updating dataset cfg for training iteration.")

    cfg_dataset_path = path_manager.get_path("path_cfg_dataset")
    with open(cfg_dataset_path, 'r') as cfg_file:
        cfg_data = yaml.safe_load(cfg_file)

    # Update DATASET path
    cfg_data['DATA_PATH'] = os.path.join("..", cfg_autolabel.DATA.PROJECT.DATASET)

    if eval_on_val:
        # Update DATA_SPLIT
        cfg_data['DATA_SPLIT']['test'] = 'val'

        # Update INFO_PATH
        cfg_data['INFO_PATH']['test'] = ['kitti_infos_val.pkl']

    elif not eval_on_val:
        # Update DATA_SPLIT
        cfg_data['DATA_SPLIT']['test'] = 'pseudo_label'

        # Update INFO_PATH
        cfg_data['INFO_PATH']['test'] = ['kitti_infos_pseudo_label.pkl']
    else:
        raise ValueError("Parameter eval_on_val in cfg_dataset_update is not valid.")
        exit()


    with open(cfg_dataset_path, 'w') as updated_cfg_file:
        yaml.dump(cfg_data, updated_cfg_file)

    if print_cfg:
        with open(cfg_dataset_path, 'r') as cfg_file:
            cfg_data = yaml.safe_load(cfg_file)
            print(yaml.dump(cfg_data))

    return

# Function that predicts objects having a trained model and a set of unlabeled data.
def predict_objects(working_path, path_manager, model):
    print("-> Predicting objects.")

    # check the model folder for validity.
    path_cfg_file_model = os.path.join(path_manager.get_path("path_cfg_models"), (model + ".yaml"))
    path_ckpt_dir = os.path.join(path_manager.get_path("path_model_ckpt_dir"), model)

    model_ckpt = os.listdir(path_ckpt_dir)
    if len(model_ckpt) != 1:
        raise ValueError("There should be exactly one file in the directory: ", path_ckpt_dir)
    if len(model_ckpt) == 0:
        raise ValueError("There is no model file in the directory: ", path_ckpt_dir)
    model_ckpt = model_ckpt[0]

    if cfg_autolabel.PIPELINE.PRINT_INFORMATION:
        print("path_cfg_file_model: ", path_cfg_file_model)
        print("path_ckpt_dir: ", path_ckpt_dir)
        print("model_ckpt: ", model_ckpt)

    # Define working paths.
    predict_objects_path = os.path.join(working_path, "autolabel_pipeline")
    predict_objects_script_path = os.path.join(predict_objects_path, "predict_objects.py")
    os.chdir(predict_objects_path)
    print("Current working path: ", predict_objects_path)

    # Trigger predict objects script.
    command = f"python {predict_objects_script_path} --cfg_file {path_cfg_file_model} --ckpt_dir {path_ckpt_dir} --ckpt {model_ckpt}"
    try:
        subprocess.run(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing the script: {e}")

    return

# Function that triggers the evaluate_labels script with according parameters.
def evaluation_metrics(cfg_autolabel, path_manager, eval_on_val):
    print("-> Evaluation metrics against ground truth.")

    folder_1 = cfg_autolabel.DATA.PATH_GROUND_TRUTHS
    if cfg_autolabel.PIPELINE.VOTING_SCHEME == "MAJORITY":
        path_pseudo_labels = path_manager.get_path("path_pseudo_labels_majority")
    elif cfg_autolabel.PIPELINE.VOTING_SCHEME == "NMS":
        path_pseudo_labels = path_manager.get_path("path_pseudo_labels_nms")
    else:
        raise ValueError("Autolabel cfg_autolabel.PIPELINE.VOTING_SCHEME is not valid. Please check.")

    if eval_on_val:
        list_to_evaluate = [path_manager.get_path("path_ptrcnn_predictions"),
                            path_manager.get_path("path_ptpillar_predictions"),
                            path_manager.get_path("path_second_predictions"),
                            path_pseudo_labels]
    else:
        list_to_evaluate = [path_pseudo_labels]

    for folder_2 in list_to_evaluate:
        print("folder_1: ", folder_1)
        print("folder_2:", folder_2)
        ret_dict, mAP3d_R40, my_eval_dict = main_evaluate_labels(cfg_autolabel, path_manager, folder_1, folder_2)

    return

# Function that converts pseudo-labels to training format.
def convert_pseudo_labels_to_labels(cfg, path_manager):
    print("-> Converting pseudo-labels to kitti label format.")

    # initial path
    if cfg.PIPELINE.VOTING_SCHEME == 'MAJORITY':
        initial_path = path_manager.get_path("path_pseudo_labels_majority")
        goal_path = pathlib.Path(os.path.join(path_manager.get_path("path_output_labels"), "majority_voting"))

    elif cfg.DATA.PIPELINE.VOTING_SCHEME == 'NMS':
        initial_path = ath_manager.get_path("path_pseudo_labels_nms")
        goal_path = pathlib.Path(os.path.join(path_manager.get_path("path_output_labels"), "nms_voting"))
    else:
        raise ValueError("DATA.PIPELINE.VOTING_SCHEME is not valid.")

   # goal path
    goal_path.mkdir(parents=True, exist_ok=True)
    if any(goal_path.iterdir()):
        shutil.rmtree(goal_path)
        goal_path.mkdir(parents=True, exist_ok=True)

    # Update the file content and remove empty files.
    print("-> Update train set with pseudo-labels. (Removing empty pseudo-labels.)")
    for csvfile in os.listdir(initial_path):
        if csvfile.endswith(".csv"):
            file_path = os.path.join(initial_path, csvfile)
            if os.path.getsize(file_path) > 0:
                frame_id = os.path.splitext(csvfile)[0]
                convert_pseudo_label_to_kitti_label(file_path, frame_id, goal_path)

    # Count the number of files after update.
    count_file_number_in_folder(goal_path)

    return

# Function that converts the pseudo-labels to match kitti training input.
def convert_pseudo_label_to_kitti_label(label_path, frame_id, goal_path):
    goal_path = os.path.join(goal_path, (frame_id + ".txt"))

    # get_label_annos returns the correct label format:
    # label, truncated, occluded, alpha, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, dim_height, dim_width, dim_length,
    # loc_x, loc_y, loc_z, rotation_y. --> Score not relevant for train.
    annotations = get_label_anno(label_path, "pseudo_labels", frame_id)

    # Write to txt.
    txt_data = []
    for i in range(len(annotations['name'])):
        line = f"{annotations['name'][i]} " \
               f"{annotations['truncated'][i]} " \
               f"{annotations['occluded'][i]} " \
               f"{annotations['alpha'][i]} " \
               f"{annotations['bbox'][i][0]} " \
               f"{annotations['bbox'][i][1]} " \
               f"{annotations['bbox'][i][2]} " \
               f"{annotations['bbox'][i][3]} " \
               f"{annotations['dimensions'][i][1]} " \
               f"{annotations['dimensions'][i][2]} " \
               f"{annotations['dimensions'][i][0]} " \
               f"{annotations['location'][i][0]} " \
               f"{annotations['location'][i][1]} " \
               f"{annotations['location'][i][2]} " \
               f"{annotations['rotation_y'][i]}"
        txt_data.append(line)

    with open(goal_path, mode='w') as txt_file:
        txt_file.write('\n'.join(txt_data))

    return

# Function that makes a backup of the originally available training data to avoid overwriting it with pseudo-labels.
def backup_original_training_data(path_manager):
    print("-> Backup the original label and train info. ")

    path_dataset = path_manager.get_path("path_project_dataset")
    path_dataset_backup = pathlib.Path(os.path.join(path_dataset, "original"))

    if os.path.exists(path_dataset_backup):
        print("Backup of ImageSets and labels already exists.")
        return

    else:
        print("Generating ImageSet and label backup.")
        path_dataset_backup.mkdir(parents=True, exist_ok=True)
        #shutil.copy(source_path, destination_path)
        shutil.copy(os.path.join(path_dataset, "ImageSets_KITTI_full/train.txt"), path_dataset_backup)
        shutil.copytree(os.path.join(path_dataset, "training/label_2"), os.path.join(path_dataset_backup, "label_2"))
        print("Backup generated.")
        return

# Function that updates the train set with available pseudo-labels.
def update_trainset(path_manager, include_og_train_data):
    print("-> Updating the trainset (ImageSets + labels)")

    # Get correct paths.
    if cfg_autolabel.PIPELINE.VOTING_SCHEME == "MAJORITY":
        path_output_labels = os.path.join(path_manager.get_path("path_output_labels"), "majority_voting")
    elif cfg_autolabel.PIPELINE.VOTING_SCHEME == "NMS":
        path_output_labels = os.path.join(path_manager.get_path("path_output_labels"), "nms_voting")
    else:
        raise ValueError("cfg_autolabel.PIPELINE.VOTING_SCHEME is not valid.")

    # Gather pseudo-label frames.
    list_labels_autolabel = []
    for filename in os.listdir(path_output_labels):
        if filename.endswith(".txt"):
            number = filename.split(".")[0]
            list_labels_autolabel.append(number)

    # Gather OG train frames if used.
    list_labels_og = []
    if include_og_train_data:
        path_output_labels_text_og = os.path.join(path_manager.get_path("path_project_dataset"), "original/train.txt")
        with open(path_output_labels_text_og, 'r') as file:
            lines = file.readlines()
            list_labels_og = [line.strip() for line in lines]

    # Case OG + pseudo.
    if include_og_train_data and not (len(list_labels_autolabel) == 0):
        list_train = list_labels_og + list_labels_autolabel
        print(f"Number of pseudo-labeled frames available for training: {len(list_labels_autolabel)}")
        print(f"Number of original frames available for training: {len(list_labels_og)}")
        print(f"-> Train set utilizes both pseudo and originally labeled frames. Total: {len(list_train)}")

    # Pseudo-labels only.
    elif not include_og_train_data and not (len(list_labels_autolabel) == 0):
            list_train = list_labels_autolabel
            print("-> Train set utilizes pseudo labeled frames only. Total: ", len(list_train))

    # Case no pseudo-labels existing.
    elif include_og_train_data and (len(list_labels_autolabel) == 0):
        print("No pseudo-labels are output with this parameter configuration. Recheck parameters.")
        exit()

    # Case nothing
    elif not include_og_train_data and (len(list_labels_autolabel) == 0):
        print("Settings defined to not use original training data. "
            "No pseudo-labels are output with this parameter configuration. Recheck parameters.")
        exit()

    # Write to .txt file
    path_imagesets_train = os.path.join(path_manager.get_path("path_project_dataset"), "ImageSets_KITTI_full/train.txt")
    list_train_sorted = sorted(list_train, key=lambda x: int(x))
    with open(path_imagesets_train, 'w') as file:
        for item in list_train_sorted:
            file.write(item + '\n')
    if cfg_autolabel.PIPELINE.PRINT_INFORMATION:
        print("-> train.txt updated.")

    # Reset label files to original kitti-og files.
    path_label_2_folder = os.path.join(path_manager.get_path("path_project_dataset"), "training/label_2")
    path_og_label_folder = os.path.join(path_manager.get_path("path_project_dataset"), "original/label_2")
    shutil.rmtree(path_label_2_folder)
    os.makedirs(path_label_2_folder)
    for filename in os.listdir(path_og_label_folder):
        source_file = os.path.join(path_og_label_folder, filename)
        target_file = os.path.join(path_label_2_folder, filename)
        shutil.copy(source_file, target_file)

    # Update label files by overwriting the pseudo-labeled frame labels.
    for filename in os.listdir(path_output_labels):
        path_label_2_file = os.path.join(path_label_2_folder, filename)
        path_output_labels_file = os.path.join(path_output_labels, filename)
        shutil.copy(path_output_labels_file, path_label_2_file)
    if cfg_autolabel.PIPELINE.PRINT_INFORMATION:
        print("-> /label_2 file updated.")

    return

# Function that creates the autolabel training .pkl files necessary for training.
def create_autolabel_training_infos(path_manager, cfg_autolabel):
    print("-> Creating training .pkl files.")

    # Remove old files:
    # FOR CREATE AUTOLABEL INFOS:
    directory_to_clean = path_manager.get_path("path_project_dataset")
    for item in os.listdir(directory_to_clean):
        item_path = os.path.join(directory_to_clean, item)
        if os.path.isfile(item_path) and any(item.lower().endswith(ext) for ext in [".pkl"]):
            os.remove(item_path)

    # Remove subfolder "gt_database" if it exists
    gt_database_path = os.path.join(directory_to_clean, "gt_database")
    if os.path.exists(gt_database_path) and os.path.isdir(gt_database_path):
        shutil.rmtree(gt_database_path)

    # trigger create_autolabel_infos directly.
    try:
        with open(path_manager.get_path("path_cfg_dataset"), 'r') as file:
            dataset_cfg = EasyDict(yaml.safe_load(file))
    except Exception as e:
        print("Error loading YAML file:", e)

    # get infos from cfg_autolabel.
    create_autolabel_infos(dataset_cfg=dataset_cfg,
                           class_names=cfg_autolabel.PIPELINE.CLASSES,
                           data_path=pathlib.Path(path_manager.get_path("path_project_dataset")),
                           save_path=pathlib.Path(path_manager.get_path("path_project_dataset")))

    return



# MODE 1
def mode_1():
    model_yaml = "pointrcnn.yaml"

    print("Processing model: ", model_yaml)

    # To be moved to yaml.
    model_file_pointrcnn = "checkpoint_epoch_75.pth"
    origin_path_pointrcnn = "/home/output/home/tools/cfgs/autolabel_models/pointrcnn/default/ckpt"
    goal_path_pointrcnn = "/home/autolabel_data/autolabel/models/pointrcnn"
    model_file_pointpillar = "checkpoint_epoch_75.pth"
    origin_path_pointpillar = "/home/output/home/tools/cfgs/autolabel_models/pointpillar/default/ckpt"
    goal_path_pointpillar = "/home/autolabel_data/autolabel/models/pointpillar"
    model_file_second = "checkpoint_epoch_72.pth"
    origin_path_second = "/home/output/home/tools/cfgs/autolabel_models/second/default/ckpt"
    goal_path_second = "/home/autolabel_data/autolabel/models/second"
    FLAG_TRAIN = False
    FLAG_COPY_MODELS = False
    print("--> Semi-supervised pseudo-labeling pipeline. Initial training.")

    if FLAG_TRAIN:
        cfg_train_model_update(path_cfg_model, print_cfg=False)
        model_training(working_path, path_cfg_model)

    if FLAG_COPY_MODELS:
        copy_pth_file(model_file_pointrcnn, origin_path_pointrcnn, goal_path_pointrcnn)
        copy_pth_file(model_file_pointpillar, origin_path_pointpillar, goal_path_pointpillar)
        copy_pth_file(model_file_second, origin_path_second, goal_path_second)
    return

# MODE 2
def mode_2(cfg_autolabel, path_manager, models, EVAL_ON_KITTI_VAL):
    # Set up pipeline for evaluation on kitti eval dataset.
    def mode_2_eval_on_val(cfg_autolabel, path_manager, models):

        FLAG_RESET_PSEUDO_LABEL_FOLDERS = True
        FLAG_PREDICT_OBJECTS = True
        FLAG_VOTE_PSEUDO_LABELS = True
        FLAG_COMPUTE_EVALUATION_METRICS = True

        print("--> Semi-supervised pseudo-labeling pipeline. Mode: Loop. EVAL ON KITTI VALIDATION SET")
        print(f"Project selected: {cfg_autolabel.DATA.PROJECT.AUTOLABEL_DATA}")

        # Reset autolabel_data folders
        if FLAG_RESET_PSEUDO_LABEL_FOLDERS:
            reset_mode2_folders(cfg_autolabel, path_manager, models)

        # cfg update + predictions
        cfg_dataset_update(cfg_autolabel, path_manager, print_cfg=True, eval_on_val=True)

        # predict proposals from models
        if FLAG_PREDICT_OBJECTS:
            # Predict the objects in frames using the pre-trained models.
            for model in models:
                print("-> predict objects for: ", model)
                predict_objects(working_path, path_manager, model)

        # Vote pseudo-labels from val set
        if FLAG_VOTE_PSEUDO_LABELS:
            main_pseudo_label(cfg_autolabel, 10000, False, 0)

        # Compute performance metrics
        if FLAG_COMPUTE_EVALUATION_METRICS:
            evaluation_metrics(cfg_autolabel, path_manager, eval_on_val=True)

        return

    # Set up pipeline for training iteration
    def mode_2_pseudo_label_iteration(cfg_autolabel, path_manager, models):

        FLAG_RESET_PSEUDO_LABEL_FOLDERS = True
        FLAG_PREDICT_OBJECTS = True
        FLAG_VOTE_PSEUDO_LABELS = True
        FLAG_COMPUTE_EVALUATION_METRICS = False
        FLAG_CONVERT_PSEUDO_LABELS = False
        FLAG_BACKUP_OG_TRAIN = False
        FLAG_UPDATE_TRAINSET = False
        FLAG_CREATE_AUTOLABEL_INFOS = False
        FLAG_TRAIN = False

        # Prerequisite: /home/autolabel_data/autolabel/models contains the selected models for this iteration.
        print("--> Semi-supervised pseudo-labeling pipeline. Mode: Loop.")
        print(f"Project selected: {cfg_autolabel.DATA.PROJECT.AUTOLABEL_DATA}")

        if FLAG_RESET_PSEUDO_LABEL_FOLDERS:
            reset_mode2_folders(cfg_autolabel, path_manager, models)

        if FLAG_PREDICT_OBJECTS:
            # Update cfg_dataset to predict pseudo-label Imageset.
            cfg_dataset_update(cfg_autolabel, path_manager, print_cfg=True,  eval_on_val=False)
            # Predict the objects in frames using the pre-trained models.
            for model in models:
                print("-> predict objects for: ", model)
                predict_objects(working_path, path_manager, model)

        if FLAG_VOTE_PSEUDO_LABELS:
            main_pseudo_label(cfg_autolabel, 10000, False, 0)

        if FLAG_COMPUTE_EVALUATION_METRICS:
            if cfg_autolabel.PIPELINE.COMPUTE_EVALUATION_METRICS:
                # cfg_dataset_update(cfg_autolabel, path_manager, print_cfg=False,  eval_on_val=False)
                evaluation_metrics(cfg_autolabel, path_manager, eval_on_val=False)

        if FLAG_CONVERT_PSEUDO_LABELS:
            convert_pseudo_labels_to_labels(cfg_autolabel, path_manager)

        if FLAG_BACKUP_OG_TRAIN:
            backup_original_training_data(path_manager)

        if FLAG_UPDATE_TRAINSET:
            update_trainset(path_manager, include_og_train_data=True)

        if FLAG_CREATE_AUTOLABEL_INFOS:
            #cfg_dataset_update(cfg_autolabel, path_manager, print_cfg=True,  eval_on_val=False)
            create_autolabel_training_infos(path_manager, cfg_autolabel)

        if FLAG_TRAIN:
            for model in cfg_autolabel.DATA.PROJECT.MODELS:
                print("Training model: ", model)
                cfg_dataset_update(cfg_autolabel, path_manager, print_cfg=False,  eval_on_val=False)
                cfg_train_model_update(path_manager, model, print_cfg=False)
                model_training(working_path, path_manager, model)
        return

    ######################
    # MAIN mode_2
    FLAG_PREPARE_AUTOLABEL_DATA_MODELS_FOLDER = False
    if FLAG_PREPARE_AUTOLABEL_DATA_MODELS_FOLDER:
        prepare_autolabel_data_models_folder(cfg_autolabel, path_manager)

    # If EVAL_ON_KITTI_VAL == True: Compute performance metrics on kitti val.
    if EVAL_ON_KITTI_VAL:
        mode_2_eval_on_val(cfg_autolabel, path_manager, models)

    if not EVAL_ON_KITTI_VAL:
        mode_2_pseudo_label_iteration(cfg_autolabel, path_manager, models)

    return



# 2 modes: Initial train and loop. MODE_INITIAL_TRAIN = True --> MODE 1
MODE_INITIAL_TRAIN = False
EVAL_ON_KITTI_VAL = True
if __name__ == "__main__":

    # Load EasyDict to access autolabel parameters.
    cfg_autolabel = load_config()
    # Load path manager to access paths easily.
    path_manager = autolabel_path_manager(cfg_autolabel)

    if cfg_autolabel.PIPELINE.PRINT_INFORMATION:
        print(f"Initial working path: {working_path}")

    # Perform initial training loop on a predefined dataset.
    if MODE_INITIAL_TRAIN:
        mode_1()

    # From selected model checkpoints, perform one semi-supervised auto-labeling loop.
    if not MODE_INITIAL_TRAIN:
        mode_2(cfg_autolabel, path_manager, cfg_autolabel.DATA.PROJECT.MODELS, EVAL_ON_KITTI_VAL)