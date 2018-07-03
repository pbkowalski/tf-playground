
dir_name = 'autoencoder_out_pipits_2'
prefixes = ['ref_eval_loss_', 'kassios_loss_']
output_name = 'out.csv'

from os import listdir # get list of all the files in a directory
from os.path import isfile, join # to manupulate file paths
import matplotlib.pyplot as plt
import numpy as np
import csv
testlist = []
listOfLists = [ [] for i in range(len(prefixes)) ]
for filename in listdir(dir_name):
    if isfile(join(dir_name, filename)):
        tokens = filename.split('.')
        if tokens[-1] == 'csv':
            for i in range(0, len(prefixes)):
                if filename.startswith(prefixes[i]):
                    with open(join(dir_name, filename), newline='') as csvfile:
                        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                        for row in reader:
                            print(row[0])
                            listOfLists[i].append(row[0])
with open(join(dir_name,output_name), 'w', newline='') as csvfile_out:
    output_writer = csv.writer(csvfile_out, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range (0, len(prefixes)):
        output_writer.writerow([prefixes[i]] + listOfLists[i])
plt.figure(num = 3)
nploss = np.array(listOfLists[0])
plt.hist(nploss, bins = 'auto', alpha = 0.5, label = 'reference images (evaluation set)')
nploss2 = np.array(listOfLists[1])
plt.hist(nploss2, bins = 'auto', alpha = 0.5, label = 'kasios images')
plt.legend(loc='upper right')

plt.savefig(join(dir_name,'joint_histogram.png'))
