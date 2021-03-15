### Implementing Jiang&Yin's algorithm
def alg_1(Ns):
    index_string_list = []
    for i in range(1,Ns+1):
        index_string_list.append(i)
    sis = index_string_list
    i = Ns
    j = i + 1
    si = sis.index(Ns)
    Nsis = len(sis)
    loop =1
    while i != j:
        inter_sis = list(zip(sis, sis[1:] + sis[:0]))
        if j > (Ns):
            j = 1
        elif (i,j) not in inter_sis and (j,i) not in inter_sis:
            sis.append(j)
            Nsis += 1
            i = j
            j = i + 1
        else:
            j = j+1
    print(sis,i,j,si,Nsis)
    return sis
