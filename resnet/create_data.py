
import csv
import os
import shutil
import random

def read_from_csv(filename):
    fields = [] 
    rows = []
    with open(filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        fields = next(csv_reader) 
        for row in csv_reader: 
            rows.append(row)
    return fields, rows

# Setup
directories = ['train', 'test', 'validate']
distributions = [0.8, 0.2, 0.0]
categories = ['reservoirs', 'terrains']
category_labels = ['1', '0']

# Copy images into folders:
for d in directories:
    if (os.path.isdir(d)):
        shutil.rmtree(d)
    for c in categories:
        new_dir = d + '/' + c
        os.makedirs(new_dir)

# Get data:
csv_fields, csv_rows = read_from_csv('_all_data/class.csv')
categories_data = {}

for i in category_labels:
    if i not in categories_data:
        categories_data[i] = []
    for row in csv_rows:
        if (row[2] == i):
           categories_data[i].append(row)

# Copy files to created directories:
directories_data = {}
random.seed(1)
random_list_dict = {}
dist_start = 0

for i in category_labels:
    random_list_dict[i] = random.sample(range(len(categories_data[i])), len(categories_data[i]))

for i in range(len(directories)):
    dir = directories[i]
    dist = distributions[i]
    for j in range(len(categories)):
        cat = categories[j]
        label = category_labels[j]
        category_n = dist_start + int(len(categories_data[label]) * dist)
        directories_data[label] = random_list_dict[label][dist_start:category_n]
        for k in directories_data[label]:
            filename = categories_data[label][k][1]
            shutil.copyfile('_all_data/' + filename , dir + '/' + cat + '/' + filename)
    dist_start = category_n