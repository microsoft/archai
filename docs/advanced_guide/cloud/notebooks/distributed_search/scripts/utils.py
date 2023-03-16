# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import json
from IPython.display import display, Image
from shutil import copyfile, rmtree


def get_results(store, blob_path, output_folder):
    """ Fetch the pareto fully trained models and show the results """
    os.makedirs(blob_path, exist_ok=True)
    store.download(blob_path, output_folder)


def download_models(store, blob_folder, output_folder, models):
    for id in models:
        sub_folder = os.path.join(output_folder, id)
        os.makedirs(sub_folder, exist_ok=True)
        store.download(os.path.join(blob_folder, id), sub_folder)


def show_results(output_folder):
    for name in os.listdir(output_folder):
        if name.endswith(".png"):
            display(Image(filename=os.path.join(output_folder, name)))


def download_best_models(store, blob_folder, output_folder):
    results_file = os.path.join(output_folder, "results.json")
    if os.path.isfile(results_file):
        best_models = json.load(open(results_file, "r"))
        for model in best_models:
            print(f"{model}\t{model['archid']}\t{model['val_acc']}")

        download_models(store, blob_folder, output_folder, list(best_models.keys()))

def copy_code_folder(self):
    """ Copies the code folder into a separate folder.  This is needed otherwise the pipeline will fail with
    UserError: The code snapshot was modified in blob storage, which could indicate tampering.
    If this was unintended, you can create a new snapshot for the run. To do so, edit any
    content in the local source directory and resubmit the run.
    """
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = 'temp_code'
    if os.path.isdir(code_dir):
        rmtree(code_dir)  # make sure old files are gone!
    os.makedirs(code_dir)
    for file in os.listdir(scripts_dir):
        path = os.path.join(scripts_dir, file)
        if os.path.isfile(path) and file.endswith('.py'):
            print(f"copying source file : {path} to {code_dir}")
            copyfile(path, os.path.join(code_dir, file))
    return code_dir