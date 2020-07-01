group = {' ': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 1, 'f': 5, 'g': 3, 'h': 6, 'i': 7, 'j': 8, 'k': 3,
         'l': 4, 'm': 9, 'n': 9, 'o': 10, 'p': 2, 'q': 3, 'r': 11, 's': 12, 't': 4, 'u': 10, 'v': 5,
         'w': 10, 'x': 3, 'y': 8, 'z': 12, '\'': 0, '-': 0}

dis_group = [[0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [2, 1, 4, 4, 4, 4, 4, 3, 4, 4, 3, 3, 4],
             [2, 4, 1, 3, 3, 4, 3, 4, 4, 4, 4, 4, 4],
             [2, 4, 3, 1, 2, 4, 3, 4, 4, 4, 4, 4, 4],
             [2, 4, 3, 2, 1, 4, 3, 4, 3, 4, 4, 3, 4],
             [2, 4, 4, 4, 4, 1, 2, 4, 3, 2, 3, 3, 4],
             [2, 4, 3, 3, 3, 2, 1, 3, 4, 4, 4, 2, 4],
             [2, 3, 4, 4, 4, 4, 3, 1, 2, 4, 4, 4, 3],
             [2, 4, 4, 4, 3, 3, 4, 2, 1, 4, 4, 2, 4],
             [2, 4, 4, 4, 4, 2, 4, 4, 4, 1, 4, 3, 4],
             [2, 3, 4, 4, 4, 3, 4, 4, 4, 4, 1, 4, 4],
             [2, 3, 4, 4, 3, 3, 2, 4, 2, 3, 4, 1, 4],
             [2, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 1]]


def dis_char(ch1, ch2):
    if ch1 == ch2:
        return 0
    return dis_group[group[ch1]][group[ch2]]


def dis_string(str1, str2):
    m = len(str1)
    n = len(str2)
    if m * n == 0:
        return 2 * (m + n)
    str1 = "0" + str1
    str2 = "0" + str2
    dis = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    dis[1][1] = dis_char(str1[1], str2[1])
    for j in range(2, n + 1):
        dis[1][j] = min(dis[1][j - 1] + 2, 2 * (j - 1) + dis_char(str1[1], str2[j]))

    for i in range(2, m + 1):
        dis[i][1] = min(dis[i - 1][1] + 2, 2 * (i - 1) + dis_char(str1[i], str2[1]))
        for j in range(2, n + 1):
            dis[i][j] = dis[i][j - 1] + 2
            for left in range(1, i):
                k_left = i - left
                min_part = 4
                for k_right in range(k_left + 1, i + 1):
                    min_part = min(min_part, dis_char(str1[k_right], str2[j]))
                dis[i][j] = min(dis[i][j], min_part + dis[k_left][j - 1] + 2 * (i - k_left - 1))
            dis[i][j] = min(dis[i][j], dis_char(str1[1], str2[j]) + 2 * (i + j - 1))
    return dis[m][n]


print(dis_string("studio", "curious"))
