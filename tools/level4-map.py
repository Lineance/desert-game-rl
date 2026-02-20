EDGES = []

for j in range(1,21,5):
    for i in range(j,j+4):
        EDGES.append((i,i+5))
        EDGES.append((i,i+1))

    EDGES.append((j+4,j+9))

EDGES = sorted(EDGES)

print(EDGES)
