import argparse
import json
import os


def hide_cells(filename):
    changed = False
    with open(filename, 'r') as f:
        data = json.load(f)
    # find all cells, and the "metadata"
    for cell in data["cells"]:
        if 'cell_type' in cell and cell['cell_type'] == 'code':
            if "metadata" in cell:
                # add the "hide" tag to the metadata
                metadata = cell["metadata"]
                if 'nbsphinx' in metadata and metadata['nbsphinx'] == 'hidden':
                    continue
                metadata["nbsphinx"] = "hidden"
                changed = True
    if changed:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help="The name of the jupyter notebook")
    args = parser.parse_args()
    file = args.name
    if os.path.splitext(file)[-1] != ".ipynb":
        print("Not a jupyter notebook")
    else:
        hide_cells(file)


if __name__ == "__main__":
    main()
