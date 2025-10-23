import regex as re
from collections import Counter, defaultdict


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# text = "this is a cat, is a cat"
# gen = [m.group() for m in re.finditer(PAT, text)]
# print(gen)
# print(type(gen))
# print(Counter(gen))


# l = re.findall(PAT, "This is very interesting, and I like it!")
# l = re.findall(PAT, text)
# print(l);
# print(type(l))
# print(Counter(l))

# a = "a"
# print(a)
# print(type(a))
# byte_str = b"a"
# print(byte_str)
# print(type(byte_str))
# print(list(byte_str))

# s = "China"
# s_encoded = s.encode("utf-8")
# print(s)
# print(s_encoded)



def train_bpe_naive(text: str, num_merges: int) -> tuple[ dict[int, bytes], list[tuple[bytes, bytes]] ]:
    # pass 
    # 1. initialize vocab ,0-255 + special token
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = "<endoftext|>".encode("utf-8")
    
    # 2. pre-tokenization: construct the frequency table
    freq_table = Counter(m.group() for m in re.finditer(PAT, text))
    freq_table_tuple = {tuple(bytes([x]) for x in key.encode("utf-8")): value for key, value in freq_table.items() }
    print(freq_table)
    print(freq_table_tuple)


    merges = []
    # 3. merge
    for i in range(num_merges):
        # get statistics for each adjacent pair
        pair_stats = defaultdict(int)
        for key, value in freq_table_tuple.items():
            for j in range(len(key)-1):
                pair_stats[(key[j], key[j+1])] += value
        # get the most frequent pair
        best_pair1 = max(pair_stats, key=pair_stats.get)
        best_pair = max(pair_stats, key=lambda k: (pair_stats[k],k))
        print(best_pair1)
        print(best_pair)
        # append merged byte to vocab
        print(b''.join(best_pair))
        vocab[257+i] = b''.join(best_pair)

        # save the merges
        merges.append(best_pair)

        # merge in the frequency table
        freq_table_tuple = merge_pair_in_table(freq_table_tuple, best_pair)



    # 4. update vocab and merges, record merge history


    # 5. update pre-tokenization


    # 6. loop 3 - 5 until num_merges is reached



    return (vocab, merges)


def merge_pair(tup: tuple[bytes], pair: tuple[bytes]) -> tuple[bytes]:
    result = []
    i = 0
    while i < len(tup):
        if tup[i:i+2] == pair:
            result.append(b''.join(pair))
            i+=2
        else:
            result.append(tup[i])
            i+=1
    return tuple(result)


def merge_pair_in_table(table: dict[tuple[bytes], int], pair: tuple[bytes]) -> dict[tuple[bytes]]:
    return {merge_pair(key, pair): value for key, value in table.items()}

if __name__ == '__main__':
    text = "low low low lower lower widest widest newest"
    vocab, merges = train_bpe_naive(text, 5)
    print(vocab)
    # print(merges)
