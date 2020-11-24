from fast_bleu import BLEU, SelfBLEU
import numpy as np
import nltk
from nltk import ngrams
from collections import Counter
import argparse
from ngram_utils import ngrams
# nltk.download('punkt')

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

def preprocess_text(file):
        list_of_gen = []
        for line in file.readlines():
                words = nltk.word_tokenize(line)
                for word in words:
                        list_of_gen.append(word)
                file.close()

        size = len(list_of_gen) 
        idx_list = [idx + 1 for idx, val in
                enumerate(list_of_gen) if val == '|endoftext|']
        res = [list_of_gen[i: j] for i, j in
                zip([0] + idx_list, idx_list + 
                ([size] if idx_list[-1] != size else []))] [:-1]

        gen = []
        for sent in res:
                sent = remove_values_from_list(sent, '<')
                sent = remove_values_from_list(sent, '|endoftext|')
                sent = remove_values_from_list(sent, '>')
                gen.append(sent)
        
        return gen

def self_bleu(cond, uncond, weight):
        self_bleu_cond = SelfBLEU(cond, weights)
        self_bleu_uncond = SelfBLEU(uncond, weights)

        self_bleu_cond = self_bleu_cond.get_score()
        self_bleu_uncond = self_bleu_uncond.get_score()

        cond_score = {}
        for k,v in self_bleu_cond.items():
                v = sum(v) / float(len(v))
                cond_score = {k : v}

        uncond_score = {}
        for k,v in self_bleu_uncond.items():
                v = sum(v)/ float(len(v))
                uncond_score = {k : v}

        return cond_score, uncond_score

def bleu(cond, uncond, weight):
        bleu = BLEU(uncond, weights)
        score = bleu.get_score(cond)

        mean_score = {}
        for k,v in score.items():
                v = sum(v) / float(len(v))
                mean_score = {k : v}
        
        return score, mean_score


def distinct_n_sentence_level(sentence, n):
        """
        Compute distinct-N for a single sentence.
        :param sentence: a list of words.
        :param n: int, ngram.
        :return: float, the metric value.
        """
        if len(sentence) == 0:
                return 0.0  # Prevent a zero division
        distinct_ngrams = set(ngrams(sentence, n))
        # len_sentence = len(sentence.split())
        len_sentence = len(list(ngrams(sentence, n)))
        # print(len(distinct_ngrams), len(sentence), len_sentence)
        return len(distinct_ngrams) / len_sentence

def distinct_n_grams(cond, uncond, n):
        cond_sentences = []
        for word_list in cond:
                sent = ' '.join(word for word in word_list)
                cond_sentences.append(sent)

        uncond_sentences = []
        for word_list in uncond:
                sent = ' '.join(word for word in word_list)
                uncond_sentences.append(sent)
        
        cond_score = sum(distinct_n_sentence_level(sentence, n) for \
                sentence in cond_sentences) / len(cond_sentences)
        uncond_score = sum(distinct_n_sentence_level(sentence, n) for \
                sentence in uncond_sentences) / len(uncond_sentences)

        return cond_score, uncond_score

  
if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument("--metric", type=str, default="self_bleu",  help= "diversity metric" )
        parser.add_argument("--n", type=int, default=1,  help= "n-gram" )
        args = parser.parse_args()


        gedi_file = open("./generated_sentences/gedi_gen_topic_topk.txt", "r")
        gedi_gen = preprocess_text(gedi_file)
        gpt2_file = open("./generated_sentences/gpt2_gen10.txt", "r")
        gpt2_gen = preprocess_text(gpt2_file)

        # print(gedi_gen)
        print(len(gedi_gen))
        print(len(gpt2_gen))

        if args.metric == 'self_bleu':
                weights = {'unigram': (1.)}
                # weights = {'trigram': (1/3., 1/3., 1/3.)}
                # weights = {'bigram': (1/2., 1/2.)}
                score1, score2 = self_bleu(gedi_gen, gpt2_gen, weights)
        elif args.metric == 'bleu':
                weights = {'unigram': (1.)}
                # weights = {'trigram': (1/3., 1/3., 1/3.)}
                # weights = {'bigram': (1/2., 1/2.)}
                score1, score2 = bleu(gedi_gen, gpt2_gen, weights)
        elif args.metric == 'distinct_n_grams':
                score1, score2 = distinct_n_grams(gedi_gen, gpt2_gen, args.n)

        # print('gedi diversity score : ', score1)
        # print('gpt2 diversity score : ', score2)
        print('pplm bleu score with itself: ', score1)
        print('pplm bleu mean score: ', score2)
