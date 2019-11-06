import Levenshtein, json, logging, sys, traceback, time
from utils import read_file, CandidateQueryFile, NGramFile, clean_query
from es_utils import ElasticObj

def resolve_search(search):
    res, s = [], 0
    if search and search['hits']['hits']:
        for hit in search['hits']['hits']:
            res.append((hit['_source']['candidate_query'], hit['_source']['candidate_query_freq'])) ; #print(json.dumps(hit, ensure_ascii=False))
            s += hit['_source']['candidate_query_freq']
    return res, s

def customize_set(file_path):
    res = []
    try:
        res = [e.strip() for e in open(file_path, encoding='utf8').readlines() if e.strip() != '']
    except Exception as e:
        logging.warn('customize_set_err=%s' % repr(e))
    return res

#print(customize_set('./right'));exit()

class queryCorrect:
    def __init__(self):
        self.VERSION = 'query_correct_1'
        self.candidate_query = read_file(CandidateQueryFile)
        self.ngrams = read_file(NGramFile)
        self.logs = {}
        self.es_obj = ElasticObj("candidate_query")
        self.right = customize_set("./right")
        self.wrong = customize_set("./wrong")
        logging.info('init queryCorrect ok, version=[%s]' % self.VERSION)

    def on_correct_begin(self):
         logging.debug('on_correct_begin')
         self.t_begin = time.time()

    def on_correct_end(self):
         logging.debug('on_correct_end')
         phead = '[on_correct_end] | req_dict=%s | queryFreq=%s | ngramFreq=%s | conditionProb=%s | result=%s | cost=%.3fs'
         logging.info(phead % (self.logs['req_dict'],self.logs['queryFreq'],self.logs['ngramFreq'],self.logs['conditionProb'], \
             self.logs['result'],(time.time()-self.t_begin)))

    def cal_condition_prob(self, n_gram, word):
        # prob(word | n_gram) = C(n_gram + word)/C(n_gram)
        numerator, denominator = n_gram + word, n_gram
        numerator_val = self.ngrams.get(numerator, 0)
        denominator_val = self.ngrams.get(denominator, self.ngrams.get(numerator, 1e-8))
        #print("%s:%d\t%s:%d" % (numerator,numerator_val, denominator, denominator_val))
        return numerator_val / denominator_val

    def need_correct(self, text):
        condition_prob = {}
        need, query_freq, ngram_freq, condi_prob = False, False, False, False
        freq = self.candidate_query.get(text, 0)
        # query 频率判断条件
        if freq < 3:
            query_freq = True
        # ngram 频率判断条件
        ngram, ngramfreq = 2, {}
        for e in [text[j: j + ngram] for j in range(0, len(text), ngram)]:
            ngramfreq[e] = self.candidate_query.get(e, 0)
            if self.candidate_query.get(e, 0) < 1:
                ngram_freq = True
            #print("ngram: %s\tquery freq: %d\tngram freq: %d" % (e, self.candidate_query.get(e, 0), self.ngrams.get(e, 0)))
        #print("query: %s\tseg query: %s\tfreq: %d" % (text, list(jieba.cut(text)), freq))
        # ngram 条件概率判断条件
        for i in range(len(text)):
            n_gram, word = text[max(0, i - 3): i], text[i]
            prob = self.cal_condition_prob(n_gram, word)
            condition_prob[word+'|'+n_gram] = prob
            if prob < 1e-4:
                condi_prob = True
        #print("query freq: %d\nngram freq: %s\ncondition prob: %s" % (freq,json.dumps(ngramfreq,ensure_ascii=False),json.dumps(condition_prob,ensure_ascii=False)))
        self.logs["queryFreq"] = str(freq)
        self.logs["ngramFreq"] = json.dumps(ngramfreq, ensure_ascii=False)
        self.logs["conditionProb"] = json.dumps(condition_prob, ensure_ascii=False)
        if freq > 3: need = False
        elif not ngram_freq: need = False
        elif query_freq or condi_prob: need = True
        #print("query_freq: %g\tngram_freq: %g\tcondi_prob:%g\tneed: %g" % (query_freq, ngram_freq, condi_prob, need))
        return need

    def correct1(self, text):
        res, dist, debug = [], [], []
        try:
            if self.need_correct(text) or str(text) in self.wrong:
                candidate, s = resolve_search(self.es_obj.search(text))
                candidate_edit = [(e[0], e[1], round(Levenshtein.ratio(e[0], text), 3)) for e in candidate if str(e[0]) != str(text)]
                sorted_candidate = sorted(candidate_edit, key=lambda d: d[2], reverse=True) # 根据query的编辑距离排序
                #print(candidate_edit, '\n', sorted_candidate); exit()
                for q, f, edit_dist in sorted_candidate[:3]:
                    normal_freq = round(f/(s + 1e-8), 3)
                    distance = round(0.6 * edit_dist + 0.4 * normal_freq, 3)
                    dist.append((q, edit_dist, normal_freq, distance))
                    debug.append({q: "freq:"+str(f)+" |edit_dist:"+str(edit_dist)+" |normal_freq:"+str(normal_freq)+" |final_distance:"+str(distance)})
                #for e in debug: print(e)    # ********** debug **********
            else:
                dist = [(text, 1, 1, 1)]
            sorted_dist = sorted(dist, key = lambda d: d[3], reverse=True)
            #print(dist, '\n', sorted_dist); exit()
            for e in sorted_dist:
                res.append((e[0], e[3]))
        except Exception as e:
            logging.warn('correct1_err=%s' % repr(e)); print(traceback.format_exc())
            res = [(text, 0)]
        return res

    def correct(self, text):
        res, dist, debug = [], [], []
        try:
            if self.need_correct(text) or str(text) in self.wrong:
                candidate, s = resolve_search(self.es_obj.search(text))
                sorted_candidate = sorted(candidate, key=lambda d: d[1], reverse=True)  # 根据query的频率排序
                #print(candidate, '\n', sorted_candidate); exit()
                for q, f in sorted_candidate:
                    if str(q) == str(text): continue
                    edit_dist = round(Levenshtein.ratio(q, text), 3)
                    normal_freq = round(f/(s + 1e-8), 3)
                    distance = round(0.8 * edit_dist + 0.2 * normal_freq, 3)
                    dist.append((q, edit_dist, normal_freq, distance))
                    debug.append({q: "freq:"+str(f)+" |edit_dist:"+str(edit_dist)+" |normal_freq:"+str(normal_freq)+" |final_distance:"+str(distance)})
                #for e in debug: print(e)    # ********** debug **********
            else:
                dist = [(text, 1, 1, 1)]
            sorted_dist = sorted(dist, key = lambda d: d[3], reverse=True)
            #print(dist, '\n', sorted_dist); exit()
            for e in sorted_dist:
                res.append((e[0], e[3]))
        except Exception as e:
            logging.warn('correct_err=%s' % repr(e)); print(traceback.format_exc())
            res = [(text, 0)]
        return res

    def run(self, req_dict):
        self.on_correct_begin()
        self.logs['req_dict'] = json.dumps(req_dict, ensure_ascii=False)
        callback_response = {}
        try:
            q = req_dict['request']['p']['query']
            q = clean_query(q)
            res = self.correct(q)
            callback_response['correct_result'] = res
            #print(json.dumps(callback_response, ensure_ascii=False))
        except Exception as e:
            logging.warn('run_err=%s' % repr(e)); print(traceback.format_exc())
            callback_response['correct_result'] = [(q, 0)]
        self.logs['result'] = json.dumps(callback_response, ensure_ascii=False)
        self.on_correct_end()
        #print(self.logs); exit()
        return callback_response

if __name__ == '__main__':
    try: que = sys.argv[1]
    except: que = "百度后端"
    req_dict = {"header": {},"request": {"c": "", "m": "query_correct", "p": {"query": que}}}
    qc = queryCorrect()
    print(qc.correct(que))
    #print(qc.run(req_dict))


