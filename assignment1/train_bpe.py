import os
from collections import defaultdict,Counter
import regex as re
import json

'''
input:
    input_path:for text
    vocab_size
    special_token:like <endoffile>
output:
    dict[int,bytes]:new vocab
    tuple[bytes,bytes]:combine rules
'''
def train_bpe(
        input_path:str|os.PathLike,
        vocab_size:int,
        special_token:list[str],
)->tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    
    #origin vocal from 0 to 255
    vocab={i:bytes([i]) for i in range(256)}
    
    num_merges=vocab_size-256-len(special_token)

    with open(input_path,'r',encoding='utf-8') as f:
        text=f.read()

    if special_token:
        special_regex='|'.join(re.escape(t) for t in special_token)

        parts=re.split(f"({special_regex})",text)

        train_segments=[p for p in parts if p not in special_token]
    else:
        train_segments=[text]

    gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    raw_counts=Counter()
    for segment in train_segments:
        words=gpt2_pat.findall(segment)
        for word in words:
            raw_counts[tuple(bytes([b])for b in word.encode('utf-8'))]+=1

    words_list=[]
    counts_list=[]

    for word_tuple,freq in raw_counts.items():
        words_list.append(list(word_tuple))
        counts_list.append(freq)
        
    #stats:{(byteA,byteB):frequency}
    stats=defaultdict(int)
        
    #indices:{(byteA,byteB):sequency1 2?}
    indices=defaultdict(set)

    for idx,word in enumerate(words_list):
        freq=counts_list[idx]
        for i in range(len(word)-1):
            pair=(word[i],word[i+1])
            stats[pair]+=freq
            indices[pair].add(idx)

    merges=[]

    for _ in range(num_merges):
        if not stats:
            break
        

        best_pair=max(stats.items(),key=lambda x:(x[1],x[0]))[0]

        if stats[best_pair]<=0:
            break

        merges.append(best_pair)

        new_token=best_pair[0]+best_pair[1]

        relevant_indices=list(indices[best_pair])

        for idx in relevant_indices:
            word=words_list[idx]

            freq=counts_list[idx]

            i=0
            while i <len(word)-1:
                if word[i]==best_pair[0] and word[i+1]==best_pair[1]:
                    if i>0:
                        prev_pair=(word[i-1],word[i])
                        stats[prev_pair]-=freq
                        if stats[prev_pair]==0:
                            del stats[prev_pair]


                    if i<len(word)-2:
                        next_pair=(word[i+1],word[i+2])
                        stats[next_pair]-=freq
                        if stats[next_pair]==0:
                            del stats[next_pair]

                    word[i]=new_token
                    del word[i+1]

                    if i>0:
                        new_prev=(word[i-1],word[i])
                        stats[new_prev]+=freq
                        indices[new_prev].add(idx)

                    if i <len(word)-1:
                        new_next=(word[i],word[i+1])
                        stats[new_next]+=freq
                        indices[new_next].add(idx)

                else:
                    i+=1
        if best_pair in stats:
            del stats[best_pair]

        if best_pair in indices:
            del indices[best_pair]
    for pair in merges:
        new_id=len(vocab)
        vocab[new_id]=pair[0]+pair[1]
    for s_tok in special_token:
        s_bytes=s_tok.encode('utf-8')
        vocab[len(vocab)]=s_bytes

    return vocab,merges



def bytes_to_unicode():
   bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
   cs=bs[:]
   n=0
   for b in range(256):
       if b not in bs:
           bs.append(b)
           cs.append(256+n)
           n+=1
   cs=[chr(n) for n in cs]
   return dict(zip(bs,cs))

def save_tokenizer_files(vocab,merges,out_dir):
    os.makedirs(out_dir,exist_ok=True)

    byte_encoder=bytes_to_unicode()

    json_vocab={
        k:"".join(byte_encoder[b] for b in v)
        for k,v in vocab.items()
    }
    with open(os.path.join(out_dir,'vocab.json'),'w',encoding='utf-8')as f:
        json.dump(json_vocab,f,indent=4)

    with open(os.path.join(out_dir,"merge.txt"),'w',encoding='utf-8')as f:
        for p1,p2 in merges:
            s1="".join(byte_encoder[b] for b in p1)
            s2="".join(byte_encoder[b] for b in p2)
            f.write(f"{s1} {s2}\n")

def main():
    input_path='data/TinyStories-GPT4-train.txt'
    vocab_size=10000
    special_token=['<|endoftext|>']

    output_dir='data/TinyStoriesV2-GPT4-train'

    print(f"start!! vocab_size=={vocab_size}")
    vocab,merges=train_bpe(input_path=input_path,vocab_size=vocab_size,special_token=special_token)
    save_tokenizer_files(vocab,merges,output_dir)

if __name__=="__main__":
    main()