pointers = open("./pointers.csv","rb")
ptrs = dict()

for line in pointers.readlines():
    line = line.decode().strip().split(",")
    ptrs[line[0]] = (int(line[1]),int(line[2]))

pointer = ptrs['abdominal'][1]

index = open("index.index","rb")
index.seek(pointer)
print(index.readline())