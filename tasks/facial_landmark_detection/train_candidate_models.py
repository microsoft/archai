#full train all candidate models and construct the final Pareto frontier

import csv
import subprocess

pareto_archids = []
search_results = []
with open('search_results.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        search_results.append(row)
        if row['is_pareto'] == 'True':
            pareto_archids.append(row['archid'])

print(f"Models to be fully trained: {pareto_archids}")

"""Train pareto models with full training data"""
full_training_accuracy = {}
data_dir = '/data/public_face_synthetics/dataset_100000'
output_dir = '/home/wchen/public_face_landmark_experiments/04_21_2023_fullrun_after_cleanup/full_train_pareto_models'
csv_file = 'search_results.csv'
num_epochs = 100
#max_num_images = 1000

for arch_id in pareto_archids:

    print(f"Training model with arch_id: {arch_id}")
    cmd = [
        'torchrun',
        '--nproc_per_node=4',
        'train.py',
        '--data-path', data_dir,
#        '--max_num_images', str(max_num_images),
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
    full_training_accuracy[arch_id] = errors[-1]

# Merge with full_training_accuracy dictionary
merged_data = []
for row in search_results:
    arch_id = row['archid']
    if arch_id in full_training_accuracy:
        row['Full training Validation Accuracy'] = full_training_accuracy[arch_id]
    else:
        row['Full training Validation Accuracy'] = ''
    merged_data.append(row)

# Write merged data to search_results_with_accuracy.csv
fieldnames = search_results[0].keys()
with open('search_results_with_accuracy.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in merged_data:
        writer.writerow(row)
