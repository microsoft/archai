import argparse
import csv


def read_status(filename):
    header = None
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            header = [x.strip() for x in row]
            break
    result = {}
    with open(filename, 'r') as f:
        reader = csv.DictReader(f, fieldnames=header, delimiter=',')
        next(reader)  # skip the header row.
        for row in reader:
            if 'name' in row:
                key = row['name']
                result[key] = row
        return result


def compare(file1, file2):
    m1 = read_status(file1)
    m2 = read_status(file2)
    for key in m1:
        if key in m2:
            r1 = m1[key]
            r2 = m2[key]
            if 'mean' not in r1:
                print(f'model {key} in {file1} is missing: mean')
            elif 'mean' not in r2:
                print(f'model {key} is {file2} missing: mean')
            elif 'f1_1k' not in r1:
                print(f'model {key} in {file1} is missing: f1_1k')
            elif 'f1_1k' not in r2:
                print(f'model {key} is {file2} missing: f1_1k')
            else:
                print(f"{key}, {r1['mean']}, {r2['mean']}, {r1['f1_1k']}, {r2['f1_1k']}")
        else:
            print(f'model {key} NOT FOUND in {file2}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare the results from 2 status files in .csv format.')
    parser.add_argument('file1', help='The first .csv file name.')
    parser.add_argument('file2', help='The second .csv file name.')
    args = parser.parse_args()
    compare(args.file1, args.file2)
