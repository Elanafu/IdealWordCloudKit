#!/usr/bin/env python3
# Author: fy
# Date: 2020-07-28
#用baidu分词改写lhy的代码

import paddlehub as hub
import os

class TFIDF:
    def __init__(self):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.idf_file = os.path.join(cur,'data/idf.txt')
        self.user_dict = os.path.join(cur,'repository/user_dic_v4.3.txt')
        self.idf_dict, self.common_idf = self.load_idf()

    def segtext(self,text):
        lac = hub.Module(name="lac")
        if (text>'') & (not text.isspace()):
            inputs = {"text": [text]}
            results = lac.lexical_analysis(data=inputs,user_dict=self.user_dict)
            return results[0]['word'], results[0]['tag']
        else:
            return [],[]

    def build_wordsdict(self,text):
        word_dict = {}
        candi_words = []
        candi_dict = {}
        words, flags = self.segtext(text)
        for i in range(len(words)):
            if flags[i] in  ['n','v','a','nz','vn','an','TIME','PER','LOC'] and len(words[i]) > 1:
                candi_words.append(words[i])
            if words[i]  not in word_dict:
                word_dict[words[i]] = 1
            else:
                word_dict[words[i]] += 1
        count_total = sum(word_dict.values())
        for word, word_count in word_dict.items():
            if word in candi_words:
                #归一化
                candi_dict[word] = word_count/count_total
            else:
                continue

        return candi_dict

    def extract_keywords(self, text, num_keywords):
        keywords_dict = {}
        candi_dict = self.build_wordsdict(text)
        for word,word_tf in candi_dict.items():
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
    text = '''（原标题：央视独家采访：陕西榆林产妇坠楼事件在场人员还原事情经过）
    # 央视新闻客户端11月24日消息，2017年8月31日晚，在陕西省榆林市第一医院绥德院区，产妇马茸茸在待产时，从医院五楼坠亡。事发后，医院方面表示，由于家属多次拒绝剖宫产，最终导致产妇难忍疼痛跳楼。但是产妇家属却声称，曾向医生多次提出剖宫产被拒绝。
    # 事情经过究竟如何，曾引起舆论纷纷，而随着时间的推移，更多的反思也留给了我们，只有解决了这起事件中暴露出的一些问题，比如患者的医疗选择权，人们对剖宫产和顺产的认识问题等，这样的悲剧才不会再次发生。央视记者找到了等待产妇的家属，主治医生，病区主任，以及当时的两位助产师，一位实习医生，希望通过他们的讲述，更准确地还原事情经过。
    # 产妇待产时坠亡，事件有何疑点。公安机关经过调查，排除他杀可能，初步认定马茸茸为跳楼自杀身亡。马茸茸为何会在医院待产期间跳楼身亡，这让所有人的目光都聚焦到了榆林第一医院，这家在当地人心目中数一数二的大医院。
    # 就这起事件来说，如何保障患者和家属的知情权，如何让患者和医生能够多一份实质化的沟通？这就需要与之相关的法律法规更加的细化、人性化并且充满温度。用这种温度来消除孕妇对未知的恐惧，来保障医患双方的权益，迎接新生儿平安健康地来到这个世界。'''

    tfidfer = TFIDF()
    for keyword in tfidfer.extract_keywords(text, 15):
        print(keyword)

'''('GL8', 0.5120852676144542)
('别克', 0.418595123916228)
('MPV', 0.4096682140915633)
('也是', 0.30725116056867247)
('商务', 0.12449233564754386)
('车型', 0.11842063498912281)
('绕不开', 0.10241705352289082)
('国内市场', 0.10241705352289082)
('中高端', 0.10241705352289082)
('ES', 0.10241705352289082)
('统治力', 0.10241705352289082)
('非也', 0.10241705352289082)
('传祺GM8', 0.10241705352289082)
('引领者', 0.09443143364912279)
('超强', 0.08236682583736842)'''

if __name__ == '__main__':
    test()



