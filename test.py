
def get_min_window(positions):
    len1 = max([pos for position_list in positions for pos in position_list]) + 1
    len2 = len(positions)

    print(f"{len1 = }")
    print(f"{len2 = }")

    sequence = [len2] * len1
    for token_key, token_pos in enumerate(positions):
        for idx in token_pos:
            sequence[idx] = token_key

    print(f"{sequence}")

    hash_pat = ([1] * len2) + [0]
    hash_str = [0] * (len2+1)

    start = 0
    start_idx = -1
    min_len = float('inf')

    count = 0
    for j, token_key in enumerate(sequence):
        print(f"{j = }, {token_key = }")
        hash_str[token_key] += 1

        if hash_str[token_key] <= hash_pat[token_key]: count += 1

        print(f"{hash_pat = }")
        print(f"{hash_str = }")
        print(f"{count = }")

        if count == len2:
            while True:
                print("\tMinimizing window...")
                print(f"\t{start = }, {sequence[start] = }")
                if hash_str[sequence[start]] > hash_pat[sequence[start]] or sequence[start] == len2:
                    if hash_str[sequence[start]] > hash_pat[sequence[start]]:
                        hash_str[sequence[start]] -= 1
                        print(f"\t{hash_pat = }")
                        print(f"\t{hash_str = }")
                    start += 1
                else:
                    print("\tFinished minimizing window...")
                    break
                
            print(f"\t\t{start = }")
            len_window = j - start + 1
            print(f"\t\t{len_window = }")
            if min_len > len_window:
                print(f"\t\tUpdating window...")
                min_len = len_window
                start_idx = start
            else: print(f"\t\tWindow not updated...")

    print(f"{sequence}")

    window = [start_idx, start_idx+min_len-1] # both inclusive

    return window, min_len

positions = [
    [3,6,11],
    [8,10],
    [1,14,9],
    [7]
]

print(get_min_window(positions))