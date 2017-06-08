import solve
file = open('Neighborhood_Zhvi_Summary_AllHomes.csv')
dataRaw = file.read()
data = list(map(lambda e: e.split(','), dataRaw.split('\n')))
data = data[1:len(data)-1]

to_feed = []
for i in data:
    a = 0
    b = 1
    if int(i[1]) >= 300000:
        a = 1
        b = 0
    if i[4] == '':
        i[4] = 0
    if i[5] == '':
        i[5] = 0
    if i[6] == '':
        i[6] = 0
    print(i)
    i = list(map(lambda e: float(e), i))
    to_feed.append([i, [a, b]])

solve.calc(to_feed)
