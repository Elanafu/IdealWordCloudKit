#!/usr/bin/env python3
# Author: fy
# Date: 2020-07-28
#用清华分词改写lhy的代码

import os
import thulac
from pyltp import Postagger

class TFIDF:
    def __init__(self):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.idf_file = os.path.join(cur,'data/idf.txt')
        self.user_dict = os.path.join(cur,'repository/carDic_20180927.txt')
        self.pos_file = os.path.join(cur,'repository/pos.model')
        self.idf_dict, self.common_idf = self.load_idf()

    def seg_pos(self, text):
        thu4car = thulac.thulac(user_dict= self.user_dict, seg_only=True)
        postagger = Postagger()
        postagger.load(self.pos_file)
        item = thu4car.cut(text, text=True)
        words = item.split(' ')
        postag = postagger.postag(words)
        print('|'.join([w+'_'+p for w, p in zip(words, postag)]))
        return words,postag


    def bulid_wordsdict(self, text):
        word_dict = {}
        candi_words = []
        candi_dict = {}
        words, flags = self.seg_pos(text)
        for i in range(len(words)):
            #只取实体词用于关键词提取,且排除单个汉字
            if flags[i] in ['n','v','a'] and len(words[i]) > 1:
                candi_words.append(words[i])
            if words[i] not in word_dict:
                word_dict[words[i]] = 1
            else:
                word_dict[words[i]] += 1
        count_total = sum(word_dict.values())
        for word, word_count in word_dict.items():
            if word in candi_words:
                candi_dict[word] = word_count/count_total
            else:
                continue
        
        return candi_dict

    def extract_keywords(self, text, num_keywords):
        keywords_dict = {}
        candi_dict = self.bulid_wordsdict(text)
        for word, word_tf in candi_dict.items():
            word_idf = self.idf_dict.get(word,self.common_idf)
            word_tfidf = word_idf*word_tf
            keywords_dict[word] = word_tfidf
        keywords_dict = sorted(keywords_dict.items(),key=lambda  asd: asd[1], reverse=True)

        return keywords_dict[:num_keywords]

    def load_idf(self):
        idf_dict = {}
        for line in open(self.idf_file):
            word, freq = line.strip().split(' ')
            idf_dict[word] = float(freq)  
        common_idf = sum(idf_dict.values())/len(idf_dict)
        return idf_dict, common_idf

def test():
    text = '''提及商务MPV，想必不少的人一定会想到，也是绕不开的一款车，别克GL8。毕竟别克GL8好歹也是在国内市场沉浸了20年，有这个知名度也不难想象。其次也是因为别克GL8的车内空间利用率和乘坐的舒适性非常高，能满足国内绝大部分购买商务MPV车型消费者的需求，特别是在中高端MPV市场，别克GL8 ES陆尊车型更是拥有超强的统治力。那别克GL8如此威风，是不是没有对手了？非也，自主品牌中高端豪华MPV引领者——传祺GM8请求一战！'''
    '''提及_v|商务_n|MPV_ws|，_wp|想必_d|不少_m|的_u|人_n|一定_d|会_v|想到_v|，_wp|也_d|是_v|绕_v|不_d|开_v|的_u|一_m|款_q|车_n|，_wp|别克GL8_i|。_wp|毕竟_d|别克GL8_i|好歹_d|也_d|是_v|在_p|国内市场_n|沉浸_v|了_u|20_m|年_q|，_wp|有_v|这个_r|知名度_n|也不_d|难_a|想象_v|。_wp|其次_r|也_d|是_v|因为_p|别克GL8_r|的_u|车内空间利用率_n|和_c|乘坐_v|的_u|舒适性_n|非常_d|高_a|，_wp|能_v|满足_v|国内_nl|绝大部分_n|购买_v|商务_n|MPV车型_b|消费者_n|的_u|需求_n|，_wp|特别_d|是_v|在_p|中高端_j|MPV_ws|市场_n|，_wp|别克GL8_r|E_ws|S陆尊_ws|车型_n|更_d|是_v|拥有_v|超强_a|的_u|统治力_n|。_wp|那_r|别克GL8_m|如此_r|威风_a|，_wp|是_v|不是_v|没有_v|对手_n|了_u|？_wp|非_d|也_u|，_wp|自主品牌_n|中高端_nd|豪华_a|MPV_ws|引领者_n|——_wp|传祺GM8_v|请求_v|一_m|战_n|！_wp'''

    tfidfer = TFIDF()
    for keyword in tfidfer.extract_keywords(text, 15):
        print(keyword)

    '''('商务', 0.12785699336774775)
('国内市场', 0.10518508199648248)
('车内空间利用率', 0.10518508199648248)
('统治力', 0.10518508199648248)
('自主品牌', 0.10518508199648248)
('传祺GM8', 0.10518508199648248)
('引领者', 0.09698363455855855)
('超强', 0.0845929562654054)
('舒适性', 0.07985337882486486)
('知名度', 0.07772682967036035)
('沉浸', 0.0764376328808108)
('威风', 0.07556080095324325)
('乘坐', 0.07100976970702702)
('提及', 0.0707080043518018)
('豪华', 0.06825512332963964)'''

if __name__ == '__main__':
    test()