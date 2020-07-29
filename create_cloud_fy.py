#!/usr/bin/env python3
# Author: fy
# Date: 2020-07-28

import os
import matplotlib.pyplot as plt
from collections import Counter
import keywords_tfidf_by_paddle as kw_paddle
import keywords_tfidf_by_ltp as kw_ltp
import keywords_tfidf as kw_jieba
from wordcloud import WordCloud, ImageColorGenerator

class CreateWordCloud:
    def __init__(self):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.outpath = os.path.join(cur, 'output')
        self.textdir = os.path.join(cur, 'text')
        self.background = os.path.join(cur, 'background')
        self.fontpath = os.path.join(cur, 'data/simhei.ttf')
        self.limit_words = 100
        self.Keyworder =  kw_paddle.TFIDF()
        return

    '''读取本地文件进行处理'''
    def read_local_file(self, textfile):
        textpath = os.path.join(self.textdir, textfile)
        content = open(textpath).read()
        return content

    '''抽取关键词'''
    def extract_keywords(self, content, word_num=20):
        keywords_dict = {}
        keywords = self.Keyworder.extract_keywords(content,word_num)
        for key in keywords:
            word = key[0]
            value = int(key[1]*1000)
            keywords_dict[word] = value
        return keywords_dict

    '''创建关键词词云图'''
    def show_cloud(self,word_dict, max_words, picturefile,save_name):
        self.backimage = os.path.join(self.background, picturefile)
        saveimage = os.path.join(self.outpath, save_name + '.jpg')
        #读入背景图片
        background_Image = plt.imread(self.backimage)
        cloud = WordCloud(font_path=self.fontpath,
                            background_color= 'white',
                            width=800,
                            height=600,
                            max_words=max_words,
                            max_font_size=500,
                            mask=background_Image
                            )
        word_cloud = cloud.generate_from_frequencies(word_dict)
        img_colors = ImageColorGenerator(background_Image)
        word_cloud.recolor(color_func=img_colors)
        plt.imshow(word_cloud)
        plt.axis('off')
        plt.savefig(saveimage)
        # plt.show()

    '''展示关键词云图'''
    def show_keywords(self, content, picturefile, words_num=20, save_name = 'test'):
        keywords_text = self.extract_keywords(content, words_num)
        self.show_cloud(keywords_text, words_num, picturefile, save_name)
        return

    def show_main(self, content, picturefile, words_num, save_name):
        print('正在生成该文本的关键词云图.....')
        name = save_name + '-keywords'
        self.show_keywords(content, picturefile, words_num, name)
        print('已完成该文本的关键词云图.....')

    '''根据用户输入文本进行处理'''
    def show_wordcloud_input(self, content, picturefile, words_num, save_name):
        self.show_main(content, picturefile, words_num, save_name)
        return

    '''根据用户输入载入本地文本进行处理'''
    def show_wordcloud_offline(self, textfile, picturefile, words_num, save_name):
        content = self.read_local_file(textfile)
        self.show_main(content, picturefile, words_num, save_name)
        return
        

def test():
    #用户输入内容
    content = '''提及商务MPV，想必不少的人一定会想到，也是绕不开的一款车，别克GL8。毕竟别克GL8好歹也是在国内市场沉浸了20年，有这个知名度也不难想象。其次也是因为别克GL8的车内空间利用率和乘坐的舒适性非常高，能满足国内绝大部分购买商务MPV车型消费者的需求，特别是在中高端MPV市场，别克GL8 ES陆尊车型更是拥有超强的统治力。那别克GL8如此威风，是不是没有对手了？非也，自主品牌中高端豪华MPV引领者——传祺GM8请求一战！'''
    #本地文件名
    textfile = 'car.txt'
    #背景图片名称
    picturefile = 'GAC.jpg'
    #图片保存名称
    save_name = 'GM8_v2'
    words_num = 50
    handler = CreateWordCloud()
    # handler.show_wordcloud_input(content, picturefile, words_num, save_name)
    handler.show_wordcloud_offline(textfile, picturefile, words_num, save_name)

if __name__ == '__main__':
    test()