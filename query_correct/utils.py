import Levenshtein, json, re, logging, traceback
import numpy as np
from collections import Counter

Ngram = 4
CandidateQueryFile = "./data/candidate_query"
NGramFile = "./data/ngrams"

def test_levenshtein():
    texta = 'kitten'    #'艾伦 图灵传'
    textb = 'sitting'    #'艾伦•图灵传'
    print(Levenshtein.distance(texta,textb))        # 计算编辑距离
    print(Levenshtein.hamming(texta,textb))  # 计算汉明距离
    print(Levenshtein.ratio(texta,textb))           # 计算莱文斯坦比
    print(Levenshtein.jaro(texta,textb))            # 计算jaro距离
    print(Levenshtein.jaro_winkler(texta,textb))    # 计算Jaro–Winkler距离
    print(Levenshtein.distance(texta,textb))

def edit_distance(word1, word2):
    len1, len2 = len(word1), len(word2)
    dp = np.zeros((len1 + 1, len2 + 1))
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            delta = 0 if word1[i - 1] == word2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + delta, min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))
    return dp[len1][len2]

def n_gram_words(text, n_gram, return_list=False):
    # n_gram 句子的词频字典
    words, words_freq = [], dict()
    try:
        # 1 - n gram
        for i in range(1, n_gram + 1):
            words += [text[j: j + i] for j in range(len(text) - i + 1)]
        words_freq = dict(Counter(words))
    except Exception as e:
        logging.warning('n_gram_words_err=%s' % repr(e)); print(traceback.format_exc())
    if return_list: return words
    else: return words_freq

def rmPunct(line):
    line = re.sub(r"[★\n-•／［…\t」＋＆　➕＊]+", "", line)
    line = re.sub(r"[,\./;'\[\]`!@#$%\^&\*\(\)=\+<> \?:\"\{\}-]+", "", line)
    line = re.sub(r"[、\|，。《》；“”‘’；【】￥！？（）： ～]+", "", line)
    line = re.sub(r"[~/'\"\(\)\^\.\*\[\]\?\\]+", "", line)
    return line

a=rmPunct("消费者/顾客word、excel、ppt、visio、xmind")

def clean_query(query):
    query = re.sub(r"[\\/、，]+", ",", query)
    query = re.sub(r"[（]+", "(", query)
    query = re.sub(r"[）]+", ")", query)
    query = re.sub(r"[【】●|]+", " ", query)
    query = re.sub(r"[：]+", ":", query)
    query = re.sub(r"[ ~]+", " ", query)
    query = query.strip().lower()
    return query

aa=clean_query("java-eam【maximo】●●●●项目经理")

def normal_qeury(text):
    re_ch = re.compile("([\u4e00-\u9fa5])", re.S)
    re_digital = re.compile("[0-9]{2,}", re.S)
    re_valid = re.compile("[简历]", re.S)
    digital = re_digital.findall(text)
    chinese = re_ch.findall(text)
    valid = re_valid.findall(text)
    if digital and not chinese: return False
    elif len(text) > 20 or len(text) < 2 or valid: return False
    else: return True

a=normal_qeury("ppp")

names = [e.strip() for e in open('./data/names', encoding='utf8').readlines() if e.strip() != '']
def is_name(text):
    text = str(text)
    if len(text) in [1, 2, 3] and text[0] in names: return True
    else: return False
aaa=is_name("贺珊")

def read_file(file_path):
    res = {}
    with open(file_path, encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            line_seg = line.split('&')
            if len(line_seg) != 2 or not line_seg[0]: continue
            try:
                k, v = line_seg[0], int(line_seg[1])
                res[k] = v
            except Exception as e:
                logging.warning('read_file_err=%s' % repr(e)); print(traceback.format_exc())
    return res

def get_info(origi_dict, key, *args, flag=False):
    res = []
    try:
        if key not in origi_dict or type(origi_dict[key]) != type({}):
            return res
        info = origi_dict[key]
        if flag:
            for arg in args[0]:
                if arg in info and info[arg] and isinstance(info[arg], str):
                    res.append(key+'_'+arg+'@'+info[arg])
        else:
            for k, v in info.items():
                for arg in args[0]:
                    if arg in v and v[arg] and isinstance(v[arg], str):
                        res.append(key+'_'+arg+'@'+v[arg])
    except Exception as e:
        logging.warning('get_info_err=%s' % repr(e)); print(traceback.format_exc())
    return res

def resolve_dict(dict_info):
    res = []
    try:
        if not isinstance(dict_info, dict): return res
        for k, v in dict_info.items():
            edu = get_info(v, 'education', ['discipline_name', 'school_name', 'station_name', 'city'])
            work = get_info(v, 'work', ['corporation_name', 'title_name', 'industry_name', 'position_name', 'station_name', 'architecture_name', 'city'])
            certificate = get_info(v, 'certificate', ['name'])
            project = get_info(v, 'project', ['corporation_name', 'name', 'position_name'])
            language = get_info(v, 'certificate', ['name', 'level', 'certificate'])
            skill = get_info(v, 'skill', ['name'])
            basic = get_info(v, 'basic', ['expect_position_name', 'not_expect_corporation_name', 'title_name', 'nation', 'achievement', \
                    'expect_industry_name', 'resume_name', 'station_name', 'city', 'industry_name', 'degree_origin_txt', 'position_name', \
                    'expect_city_names', 'expect_type', 'corporation_type', 'corporation_name', 'degree_origin', 'discipline_name'], flag=True)
            res.extend(edu); res.extend(work); res.extend(certificate); res.extend(project); res.extend(skill); res.extend(basic)
            res.extend(language)
    except Exception as e:
        logging.warning('resolve_dict_err=%s' % repr(e)); print(traceback.format_exc())
    return res

def parse_line(line):
    querys, tmp = [], []
    try:
        line = line.strip().lower().split('\t')
        if len(line) >= 5:
            tmp.append("query@"+line[5])
        if len(line) >= 36:
            cv_info = json.loads(line[36])
            tmp.extend(resolve_dict(cv_info))
        querys = tmp
        '''
        for q in tmp:
            q = clean_query(q)
            if normal_qeury(q):
                querys.append(q)
        '''
    except Exception as e:
        logging.warning('parse_line_err=%s' % repr(e)); print(traceback.format_exc())
    return querys

def parse_line_ngrams(line):
    ngrams = []
    try:
        querys = parse_line(line)
        for q in querys:
            if not isinstance(q, str): continue
            ngrams.extend(n_gram_words(q, 4, True))
    except Exception as e:
        logging.warning('parse_line_ngrams_err=%s' % repr(e)); print(traceback.format_exc())
    return ngrams

def parse_line_querys(line):
    querys = []
    try:
        querys = parse_line(line)
    except Exception as e:
        logging.warning('parse_line_querys_err=%s' % repr(e)); print(traceback.format_exc())
    return querys

def isenglish(keyword):
    return all(ord(c) < 128 for c in keyword)

def filter(Q, query_freq, edit_dist_th=0.8, freq_th=1000):
    query_dist, sorted_query_dist = {}, []  ; a=query_freq.get(Q, 0); aa=isenglish(Q)
    if int(query_freq.get(Q, 0)) > 15 or not isenglish(Q):
        return False
    for q, f in query_freq.items():
        if q != Q and len(q) == len(Q) and Levenshtein.ratio(q, Q) > edit_dist_th:
            query_dist[q] = int(query_freq.get(q, 0))
    sorted_query_dist = sorted(query_dist.items(), key=lambda d: d[1], reverse=True)
    if sorted_query_dist and sorted_query_dist[0][1] > freq_th and query_freq.get(Q, 0) < sorted_query_dist[0][1]:
        print(Q+'->'+sorted_query_dist[0][0])
        return True
    else:
        return False

def valid_qeury_freq(line):
    q, f, invalid = '', '0', False
    line = line.strip().split("@")
    if len(line) != 2:
        return '', '0', True
    query_freq = line[1].split('&')
    if len(query_freq) != 2 or not query_freq[1].isdigit():
        return '', '0', True
    if line[0] == 'query':
        invalid = True
    if line[0] in ['basic_resume_name']:
        pass
    q, f = query_freq[0], query_freq[1]
    return q, f, invalid

def resolv_querys(file_path, freq_threshold=10, candidate_path=CandidateQueryFile, ngram_path=NGramFile):
    print("input file: %s\nfreq threshold: %d\ncandidate query file: %s\nngram file: %s" % (file_path, freq_threshold, candidate_path, ngram_path))
    candidate_query, ngram_query = [], []
    try:
        candidate, origion_querys = {}, []
        # ********** 读取统计文件得到query和频率 **********
        with open(file_path, encoding="utf8") as f:
            for line in f.readlines():
                if line.split('@')[0] == 'query': origion_querys.append(line.strip())
                q, f, invalid = valid_qeury_freq(line)
                q = clean_query(q)
                if int(f) < 10 and is_name(q): continue     # 过滤掉姓名
                if invalid or not normal_qeury(q): continue # 过滤掉无效的query
                if q not in candidate: candidate[q] = 0
                candidate[q] += int(f)
        candidate_top = {k: v for k, v in candidate.items() if v > freq_threshold}
        for e in origion_querys:
            try:
                q, f = e.split('@')[1].split('&')
                q = clean_query(q)
                if int(f) < 20 and is_name(q): continue
            except:
                continue
            if normal_qeury(q) and int(f) > 3:
                if q not in candidate_top: candidate_top[q] = 0
                candidate_top[q] += int(f)
        # ********** 清洗query和过滤频率过低的query **********
        query_result = []
        candidate_sorted = sorted(candidate_top.items(), key=lambda d: d[1], reverse=True)
        for query, freq in candidate_sorted:
            query = clean_query(query)
            if not normal_qeury(query): continue
            query_result.append((query, freq))
        # ********** 得到最终的query和ngrams集合 **********
        print("origin query: %d\ncandidate query: %d" % (len(candidate_sorted), len(query_result)))
        ngrams = {}
        for e in query_result:
            candidate_query.append(e[0] + '&' + str(e[1]))
            for w in n_gram_words(e[0], 4, True):
                if w not in ngrams: ngrams[w] = 0
                ngrams[w] += 1
        for k, v in ngrams.items():
            ngram_query.append(k + '&' + str(v))
        # ********** 写入文件 **********
        with open(candidate_path, 'w', encoding='utf8') as f:
            for e in candidate_query: f.write(e + '\n')
        with open(ngram_path, 'w', encoding='utf8') as f:
            for e in ngram_query: f.write(e + '\n')
    except Exception as e:
        logging.warning('resolv_querys_err=%s' % repr(e)); print(traceback.format_exc())

def test():
    #read_file(CandidateQueryFile)
    txt = open("../query_correct_0/data/search_data.log1", encoding="utf8").readlines()
    ngrams, querys = [], []
    for line in txt:
        ngrams.extend(parse_line_ngrams(line))
        querys.extend(parse_line_querys(line))
    a=1

if __name__ == '__main__':
#    resolv_querys("./data/querys1", 10, './data/q', './data/n'); exit()  # 构建query数据
#    resolv_querys("../candidate_query1/querys", 10, './data/q', './data/n'); exit()     # 构建query数据
    resolv_querys("../candidate_query1/querys", 10); exit()     # 构建query数据
#    resolv_querys("../candidate_query1/querys", 5, "./q", "./n"); exit()     # 构建query数据
    test(); exit()
    a = normal_qeury("k12d2d2")
    print(read_file(CandidateQueryFile)); exit()
    exit()
    s1, s2 = '人工智能行业', '智能人工'
    print('%s\t%s' % (edit_distance(s1, s2), Levenshtein.distance(s1, s2)))
    pass


