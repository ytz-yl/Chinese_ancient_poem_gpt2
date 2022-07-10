# GPT2-Chinese-poem

本项目是用来学习pytorch与transformers的文本生成项目，选择了我比较喜欢的诗词领域来试试手。



用于生成采用平水韵的诗句，五言七言、绝句律诗排律等格式可调。

本项目采用的预训练模型为[古诗词GPT-2预训练模型](https://github.com/Morizeyao/GPT2-Chinese#%E6%A8%A1%E5%9E%8B%E5%88%86%E4%BA%AB)。模型由UER-py项目训练得到，在Huggingface Model Hub可以找到。更多模型的细节请参考[gpt2-chinese-poem](https://huggingface.co/uer/gpt2-chinese-poem)。感谢作者开源。

### 生成诗句

``` bash
python ./generate.py --model_path='你的模型路径' --rhyme_path='cache/rhyme.json' --prefix='' --topic='离愁别绪' --len_sentence=5 --num_sentence=8
```
