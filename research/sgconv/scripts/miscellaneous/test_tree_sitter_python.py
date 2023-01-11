# https://pypi.org/project/tree-sitter-languages/
from datasets import load_from_disk
from tree_sitter_languages import get_language, get_parser


def main():
    _ = get_language("python")
    parser = get_parser("python")

    codestr = "num_classes = 10 np.random.seed(133) def maybe_extract(filename, force=False): root = os.path.splitext(os.path.splitext(filename)[0])[0] # remove .tar.gz if os.path.isdir(root) and not force: # You may override by setting force=True. print('%s already present - Skipping extraction of %s.' % (root, filename)) else: print('Extracting data for %s. This may take a while. Please wait.' % root) tar = tarfile.open(filename) sys.stdout.flush() tar.extractall(data_root) tar.close() data_folders = [ os.path.join(root, d) for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))] if len(data_folders) != num_classes: raise Exception( 'Expected %d folders, one per class. Found %d instead.' % ( num_classes, len(data_folders))) print(data_folders) return data_folders train_folders = maybe_extract(train_filename) test_folders = maybe_extract(test_filename)"

    # create concrete syntax tree
    # TODO: have to understand how to
    # get information out from this tree
    _ = parser.parse(codestr.encode())

    print("done.")


if __name__ == "__main__":
    main()
