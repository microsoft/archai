import matplotlib.pyplot as plt



def main():
    
    # plot spe vs. important hyperparams
    legends = ['conditionalbatch_256_freezebatch_2048_freezeepochs_5_freezethresh_0.6',\
               'conditionalbatch_256_freezebatch_2048_freezeepochs_10_freezethresh_0.6',\
               'conditionalbatch_256_freezebatch_2048_freezeepochs_5_freezethresh_0.5',\
               'conditionalbatch_256_freezebatch_2048_freezeepochs_10_freezethresh_0.5',\
               'conditionalbatch_256_freezebatch_2048_freezeepochs_5_freezethresh_0.4',\
               'conditionalbatch_256_freezebatch_2048_freezeepochs_10_freezethresh_0.4',\
               'conditionalbatch_256_freezebatch_2048_freezeepochs_5_freezethresh_0.3',\
               'conditionalbatch_256_freezebatch_2048_freezeepochs_10_freezethresh_0.3']

    spes = [0.81, 0.85, 0.71, 0.75, 0.52, 0.57, 0.43, 0.51]

    total_durations = [238.21, 276.71, 158.92, 204.53, 99.32, 123.42, 61.99, 103.48]
    total_std = [294.70, 290.37, 223.42, 208.84, 102.00, 81.14, 43.50, 45.41]

    conditional_durations = [197.05, 194.98, 111.83, 110.01, 52.41, 41.93, 20.87, 21.06]
    conditional_std = [294.86, 290.68, 223.68, 209.26, 102.05, 81.43, 43.42, 45.12]

    partial_durations = [41.16, 81.72, 47.08, 94.51, 46.91, 81.48, 41.11, 82.41]
    partial_std = [1.92, 3.2, 1.92, 4.05, 1.85, 3.74, 1.52, 3.10]

    reg_spes = [0.78, 0.9, 0.94]
    reg_total_durations = [110.98, 223.71, 379.28]

    reg_legends = ['regular_batch_1024_epochs_10', 'regular_batch_1024_epochs_20', 'regular_batch_1024_epochs_30']


    plt.scatter(total_durations[0], spes[0])
    plt.scatter(total_durations[1], spes[1])
    plt.scatter(total_durations[2], spes[2])
    plt.scatter(total_durations[3], spes[3])
    plt.scatter(total_durations[4], spes[4])
    plt.scatter(total_durations[5], spes[5])
    plt.scatter(total_durations[6], spes[6])
    plt.scatter(total_durations[7], spes[7])
    plt.scatter(reg_total_durations[0], reg_spes[0])
    plt.scatter(reg_total_durations[1], reg_spes[1])
    plt.scatter(reg_total_durations[2], reg_spes[2])
    plt.legend(legends + reg_legends)
    plt.xlabel('Avg. Wall Clock (s)')
    plt.ylabel('Spearman Correlation')
    plt.grid()
    plt.show()






if __name__ == '__main__':
    main()


