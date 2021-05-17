### Implementing Jiang&Yin's algorithm
def alg_2(SI):
    Ns = len(SI)
    index_string_list = []
    sis = []
    for i in range(1,Ns+1):
        index_string_list.append(i)
        sis.append(SI[i-1])
    i = Ns
    j = i + 1
    si = index_string_list.index(Ns-1)
    Nsis = len(index_string_list)
    loop =1
    while i != j:
        inter_sis = list(zip(index_string_list, index_string_list[1:] + index_string_list[:0]))
        if j > (Ns):
            j = 1
        elif (i,j) not in inter_sis and (j,i) not in inter_sis:
            index_string_list.append(j)
            sis.append(SI[j-1])
            Nsis += 1
            i = j
            j = i + 1
        else:
            j = j+1
    print(index_string_list,i,j,si,Nsis)
    return sis
