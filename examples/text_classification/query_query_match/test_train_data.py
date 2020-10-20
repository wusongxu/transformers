
result = {'0':0,'1':0}
with open('./dataset/dev.tsv','r') as f:
    for line in f.readlines():
        if line[0] == '0':
            result['0'] +=1
        elif line[0] == '1':
            result['1'] +=1
print(result)
print(result['0']/result['1'])