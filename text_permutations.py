import tensorflow as ts
import string
char2num=dict(zip(string.ascii_letters,[ord(c)%32 for c in string.ascii_letters]))

data = open('data_text.txt')
data_a = data.read()
data_b = data_a.split('\n')
final = []
for i in data_b:
    if len(i) > 0:
        final += [i.split(',')]

neurons = {'mean': 5}

# Training
for i in final:
    decision = 'n'
    mean = 0
    for a in i[0]:
        mean += char2num[a]
    if mean >= neurons['mean'] - 0.5 and mean <= neurons['mean'] + 0.5:
        decision = 'y'
    if decision == i[1]:
        print("Success! ", mean, " vs. ", neurons['mean'])
    else:
        if i[1] == 'y':
            neurons['mean'] += (mean - neurons['mean']) / 2
        print('Failure! ', mean, " vs. ", neurons['mean'])

while True:
    h = input('STRING> ')
    decision = 'n'
    mean = 0
    for a in i[0]:
        mean += char2num[a]
    if mean >= neurons['mean'] - 0.5 and mean <= neurons['mean'] + 0.5:
        decision = 'y'
    print(decision)
