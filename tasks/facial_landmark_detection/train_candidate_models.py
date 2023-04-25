import csv
import subprocess

"""Train the models that are in the pareto front"""

# Please change the following variables to your own path
data_dir = 'face_synthetics/dataset_100000'
output_dir = './output'
csv_file = 'search_results.csv'

# Read the search results and pick the models in the pareto front
pareto_archids = []
search_results = []
with open(csv_file, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        search_results.append(row)
        if row['is_pareto'] == 'True':
            pareto_archids.append(row['archid'])
print(f"Models to be trained: {pareto_archids}")

# Train the models with subprocess call
training_accuracy = {}
num_epochs = 100
for arch_id in pareto_archids:

    print(f"Training model with arch_id: {arch_id}")
    cmd = [
        'torchrun',
        '--nproc_per_node=4',
        'train.py',
        '--data-path', data_dir,
        '--output_dir', output_dir,
        '--search_result_archid', arch_id,
        '--search_result_csv', csv_file,
        '--train-crop-size', '128',
        '--epochs', str(num_epochs),
        '--batch-size', '32',
        '--lr', '0.001',
        '--opt', 'adamw',
        '--lr-scheduler', 'steplr',
        '--lr-step-size', '100',
        '--lr-gamma', '0.5',
        '-wd', '0.00001'
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)

    errors = []
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            if output.startswith('Test:'):
                if 'Error' in output:
                    error_str = output.split()[-1]
                    error = float(error_str)
                    errors.append(error)

    result = process.poll()
    assert errors and len(errors) != 0 #should have at least one error
    training_accuracy[arch_id] = errors[-1]

# Merge training accuracy to search_results
merged_data = []
for row in search_results:
    arch_id = row['archid']
    if arch_id in training_accuracy:
        row['Full training Validation Accuracy'] = training_accuracy[arch_id]
    else:
        row['Full training Validation Accuracy'] = ''
    merged_data.append(row)

# Write to csv
fieldnames = search_results[0].keys()
with open('search_results_with_accuracy.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in merged_data:
        writer.writerow(row)
