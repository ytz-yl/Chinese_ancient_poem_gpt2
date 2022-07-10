import json
import argparse


def remove_brackets(line):
    """去掉括号中的注释，去掉同一韵脚中相同的字"""
    new_line = []
    need_read = True
    for c in line:
        if need_read:
            if c in '[<':
                need_read = False
            elif c not in ' \n':
                new_line.append(c)
        elif c in ']>':
            need_read = True
    return list(set(new_line))


def get_rhyme_dict(rhyme_path, save_path='rhyme.json', encoding='utf-8'):
    with open(rhyme_path, 'r', encoding=encoding) as f:
        lines = f.readlines()
    rhyme_index = pingze_board_index = 0
    rhyme_names = []
    find_rhymes = []
    chr2rhyme = dict()
    for line in lines:
        if line.startswith('仄韵'):
            pingze_board_index = rhyme_index
        if len(line) < 6:
            continue
        i = max(line.find(':'), line.find('：'))
        rhyme_names.append(line[:i])
        chr_list = remove_brackets(line[i + 1:])
        for c in chr_list:
            chr2rhyme[c] = chr2rhyme.get(c, []) + [rhyme_index]
        find_rhymes.append(chr_list)
        rhyme_index += 1

    rhyme_dict = {'rhyme_names': rhyme_names,
                  'pingze_board_index': pingze_board_index,
                  'len_rhymes': rhyme_index,
                  'find_rhymes': find_rhymes,
                  'chr2rhyme': chr2rhyme,}

    with open(save_path, 'w', encoding=encoding) as f:
        f.write(json.dumps(rhyme_dict, ensure_ascii=False, indent=2))

    return rhyme_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rhyme_path', default='PingShuiRhyme.txt', type=str, required=False, help='txt韵表的读取路径')
    parser.add_argument('--save_rhyme_path', default='rhyme.json', type=str, required=False, help="保存韵表的路径")

    args = parser.parse_args()
    rhyme_path = args.rhyme_path
    save_rhyme_path = args.save_rhyme_path
    get_rhyme_dict(rhyme_path, save_path=save_rhyme_path)


if __name__ == '__main__':
    main()
