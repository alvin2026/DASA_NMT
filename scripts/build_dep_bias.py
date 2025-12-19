import argparse
import numpy as np
import stanza


def bpe_tokens_to_words(bpe_tokens, bpe_suffix="@@"):
    """
    Convert subword-nmt BPE tokens to word strings, and return sub2word mapping.
    Example: ["Hel@@","lo","world"] -> words ["Hello","world"], sub2word [0,0,1]
    """
    words = []
    sub2word = []
    cur = ""
    widx = -1

    for tok in bpe_tokens:
        if widx == -1 or cur == "":
            widx += 1
            cur = ""
            words.append("")  # placeholder

        sub2word.append(widx)

        if tok.endswith(bpe_suffix):
            cur += tok[:-len(bpe_suffix)]
            words[widx] = cur
        else:
            cur += tok
            words[widx] = cur
            cur = ""  # word ends
    # handle edge: if sentence ends with suffix (shouldn't)
    return words, sub2word


def build_tree_shortest_paths(heads):
    """
    heads: list[int] length n, 0-based head index, -1 for root
    dependency tree -> undirected adjacency -> all-pairs shortest path by BFS per node.
    """
    n = len(heads)
    adj = [[] for _ in range(n)]
    for i, h in enumerate(heads):
        if h >= 0:
            adj[i].append(h)
            adj[h].append(i)

    D = np.full((n, n), 10**9, dtype=np.int32)
    for s in range(n):
        # BFS
        q = [s]
        D[s, s] = 0
        front = 0
        while front < len(q):
            u = q[front]
            front += 1
            for v in adj[u]:
                if D[s, v] > D[s, u] + 1:
                    D[s, v] = D[s, u] + 1
                    q.append(v)
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpe-src", required=True, help="BPE'd source file, one sentence per line")
    ap.add_argument("--out-dir", required=True, help="Output dir for {idx}.npy")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--mode", choices=["mul", "log"], default="mul")
    ap.add_argument("--eps", type=float, default=1e-9)
    args = ap.parse_args()

    nlp = stanza.Pipeline(lang=args.lang, processors="tokenize,pos,lemma,depparse", tokenize_pretokenized=True)

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.bpe_src, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            bpe_tokens = line.strip().split()
            if len(bpe_tokens) == 0:
                np.save(os.path.join(args.out_dir, f"{idx}.npy"), np.ones((1,1), dtype=np.float32))
                continue

            words, sub2word = bpe_tokens_to_words(bpe_tokens)

            # stanza pretokenized expects list of sentences, each sentence is list of tokens (words)
            doc = nlp([words])
            sent = doc.sentences[0]
            # stanza head: 1..n, root head=0; convert to 0-based, root=-1
            heads = []
            for w in sent.words:
                h = w.head - 1
                heads.append(h)  # root becomes -1

            n = len(words)
            D = build_tree_shortest_paths(heads)  # [n,n]
            W = 1.0 / (D.astype(np.float32) + args.eps)  # inverse distance

            # row-normalize
            W = W / (W.sum(axis=1, keepdims=True) + args.eps)

            # lift to subword level
            T = len(bpe_tokens)
            Wsub = np.zeros((T, T), dtype=np.float32)
            for i in range(T):
                wi = sub2word[i]
                for j in range(T):
                    wj = sub2word[j]
                    Wsub[i, j] = W[wi, wj]

            np.save(os.path.join(args.out_dir, f"{idx}.npy"), Wsub)


if __name__ == "__main__":
    main()