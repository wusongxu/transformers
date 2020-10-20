import sys
import json

source_path = sys.argv[1]
aim_path = sys.argv[2]

with open('./' + source_path, 'r') as f, open('./' + aim_path, 'w') as fw:
    for line in f.readlines():
        line = json.loads(line)
        if 'label' in line.keys():
            fw.write(line['label'] + '\t' + 'ddd' + '\t' + line['sentence1'] + '\t' + line['sentence2'] + '\n')
        else:
            fw.write('tests' + '\t' + 'ddd' + '\t' + line['sentence1'] + '\t' + line['sentence2'] + '\n')
