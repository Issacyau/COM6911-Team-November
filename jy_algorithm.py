### Implementing Jiang&Yin's algorithm
def index_string(Ns):
    index_string_list = []
    for i in range(1,Ns):
        index_string_list.append(i)
    return index_string_list

def alg_1(Ns):
    sis = index_string(Ns)
    i = Ns - 1
    j = i + 1
    si = sis.index(Ns-1)
    Nsis = len(sis)
    loop =1
    while i != j:
        inter_sis = list(zip(sis, sis[1:] + sis[:0]))
        if j > (Ns-1):
            j = 1
        elif (i,j) not in inter_sis and (j,i) not in inter_sis:
            sis.append(j)
            Nsis += 1
            i = j
            j = i + 1
        else:
            j = j+1
    print(i,j,si,Nsis)
    return(sis)
alg_1(7)
print(alg_1(7))
print(alg_1(10))