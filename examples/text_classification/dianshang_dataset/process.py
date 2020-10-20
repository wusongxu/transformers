import xlrd

path = '/Users/songxuwu/Desktop/通用语料和店铺语料/data.xls'

data = xlrd.open_workbook(path)
data.sheet_names()
table = data.sheet_by_name('行业数据')
common_data = []
for row_num in range(table.nrows):
    if row_num == 0:
        continue
    row_value = table.row_values(row_num)
    common_data.append(row_value[2:4])
common_data_dict = {}
for row in common_data:
    query_list = row[1].split('/')
    if len(query_list) >= 4:
        common_data_dict[row[0]] = query_list
common_data_ = [(key, value) for key, value in common_data_dict.items()]
train_data = []

import random

for i, one_tuple in enumerate(common_data_):
    number = 0
    for j in range(len(one_tuple[1])):
        for k in range(j, len(one_tuple[1])):
            train_data.append('1' + '\t' + one_tuple[1][j] + '\t' + one_tuple[1][k] + '\n')
            number += 1
    for l in range(number * 2):
        sentence_a = one_tuple[1][random.randint(0, len(one_tuple[1]) - 1)]
        n = random.randint(0, len(common_data_) - 1)
        while n == i:
            n = random.randint(0, len(common_data_) - 1)
        sentence_b = common_data_[n][1][random.randint(0, len(common_data_[n][1]) - 1)]
        train_data.append('0' + '\t' + sentence_a + '\t' + sentence_b + '\n')

import random

random.shuffle(train_data)
train = train_data[:int(len(train_data) * 4 / 5)]
dev = train_data[int(len(train_data) * 4 / 5):]
with open('../query-query-match/train.tsv', 'w') as ft, open('../query-query-match/dev.tsv', 'w') as fd:
    for line in train:
        ft.write(line)
    for line in dev:
        fd.write(line)
