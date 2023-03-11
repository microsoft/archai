import os
import json
from IPython.display import display, Image


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
