
dump = []
X = [[1, 2, 3], [4, 5, 6]]
Y = ['a', 'b', 'c', 'd', 'e', 'f']

for x in X:
    for y in Y:
        dump.append([x+[y]][0])

print(dump)
