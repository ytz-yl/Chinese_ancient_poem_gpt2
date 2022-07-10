import argparse
import json
from model.gpt_poem import GPTPoem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='D:/NLP/models/gpt2-chinese-poem', type=str, help='模型的路径')
    parser.add_argument('--rhyme_path', default='cache/rhyme.json', type=str, help='韵表路径')
    parser.add_argument('--prefix', default='', type=str, help='诗句开头')
    parser.add_argument('--topic', default='离愁别绪', type=str, help='诗句主题')
    parser.add_argument('--len_sentence', default=5, type=int, help='五言七言')
    parser.add_argument('--num_sentence', default=8, type=int, help='绝句律诗排律')

    args = parser.parse_args()
    model_path, rhyme_path = args.model_path, args.rhyme_path
    with open(rhyme_path, 'r', encoding='utf-8') as f:
        rhyme_dict = json.load(f)

    prefix, topic = args.prefix, args.topic
    len_sentence, num_sentence = args.len_sentence, args.num_sentence

    model = GPTPoem(model_path=model_path, rhyme_path=rhyme_dict)
    poem_sample = model.generate(prefix=prefix, topic=topic, len_sentence=len_sentence, num_sentence=num_sentence)

    print(poem_sample)


if __name__ == '__main__':
    main()
