import argparse
import gzip

from tqdm import tqdm

def load_data(path, remap):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for l in f:
            l = eval(l)
            if l['asin'] in remap:
                l['asin'] = int(remap[l['asin']]) - 1
                yield l

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--path', type=str, default='meta_Sports_and_Outdoors.json.gz')
    argparser.add_argument('--id_map', type=str,default='item_list2.txt')
    args = argparser.parse_args()
    
    with open(args.id_map, 'r', encoding='utf-8') as f:
        id_map = {org: remap for (org, remap) in map(lambda x: x.split(), f)}
    
    data = tqdm(load_data(args.path, remap=id_map), desc='loading data')
    data = sorted(data, key=lambda x: x['asin'])
    
    with open('categories.txt', 'w', encoding='utf-8') as f:
        f.writelines([str(item['categories'][0]) + '\n' for item in data])
