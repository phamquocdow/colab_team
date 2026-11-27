"""Tiền xử lý văn bản dùng chung cho cả huấn luyện (notebook 3) và dự đoán bài mới (notebook 4).

Làm sạch → Tokenization → POS tagging → Stop-word removal → Lemmatization.
Dự đoán bài mới PHẢI đi qua đúng hàm `preprocess_article` này thì đặc trưng mới khớp với lúc huấn luyện.
"""
import re

import nltk
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import PerceptronTagger


def _ensure_nltk():
    """Tải tài nguyên NLTK nếu máy chưa có (chỉ tải khi thiếu)."""
    for path, name in [("corpora/stopwords", "stopwords"), ("corpora/wordnet", "wordnet"),
                       ("corpora/omw-1.4", "omw-1.4"),
                       ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


_ensure_nltk()
STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
tagger = PerceptronTagger()                     # nạp một lần (nltk.pos_tag nạp lại mô hình mỗi lần gọi)

NOISE_RE = re.compile(r"<[^>]+>|&#?\w+;|https?://\S+|www\.\S+|@\w+"                      # HTML, entity, URL, @tài khoản
                      r"|\b[\w-]+(?:\.[\w-]+)*\.(?:com|org|net|gov|rs|ly|tv|co|uk|us)/\S*"  # link không có http: bit.ly/.., tmsnrt.rs/..
                      r"|\(function\(d, s, id\).*?facebook-jssdk['\"]?\)\);?"              # mã nhúng Facebook
                      r"|//\s*<\s*!\[cdata\[", re.S | re.I)                                # "// <![CDATA["

SOURCE_PATTERNS = [
    # --- Reuters (tin True) ---
    r"^.{0,150}?\(reuters\)\s*[-–—]*\s*",                                   # "WASHINGTON (Reuters) -"
    r"\((?:additional )?(?:reporting|writing|editing|compiled)\b[^)]*\)",   # "(Reporting by ...; Editing by ...)"
    r"this article was funded in part by sap.*?(?:creation|production)\.?", # câu tài trợ SAP
    r"the following statements were posted.*?confirmed their accuracy\.?",  # mở đầu bài tổng hợp tweet
    r"source link:?|\[\s*\d{3,4}\s*est\s*\]",                               # "Source link:", "[0804 EST]"
    r"^\s*(?:factbox|exclusive|highlights)\s*:",                            # tiền tố tiêu đề
    r"\[n[a-z0-9]{6,}[\]}]",                                                # mã bài nội bộ "[nL1N1NQ15C]"
    r"\breuters\b",
    # --- Các trang tin giả (tin Fake) ---
    r"[\w'’ -]{0,40}/\s*getty ?images?",                                    # "Photo by Win McNamee/Getty Images"
    r"\bfeatured (?:image|photo)s?\b",                                      # "Featured image via ..."
    r"\b(?:image|photo|picture)s? (?:via|by|courtesy of|credit)\b",
    r"\bgetty ?images?\b|\bscreen ?(?:capture|grab|shot)s?\b",
    r"\bvia (?:youtube|twitter|facebook|flickr|instagram|wikimedia(?: commons)?)\b",
    r"support (?:our work|21wire)\W+subscribe(?: now)?\W+(?:&|and)\W+become a member\W*21wire\.tv",  # chân trang 21WIRE
    r"\b21st century wire(?: says)?\b|\b21wire(?:\.tv)?\b",
    r"(?:\b|(?<=\.tv))read more\b[^:.]{0,40}:|\bcontinue this story at\b|\b\w+ files(?=\s*support\b)",
    r"tune in to the alternate current radio network.{0,80}?boiler room|\bacr\b",  # quảng cáo radio ACR
]
SOURCE_RE = re.compile("|".join(SOURCE_PATTERNS), re.I | re.S)
POS_MAP = {"J": "a", "V": "v", "R": "r"}       # nhãn Penn Treebank → từ loại WordNet (mặc định: danh từ)
_lemma_cache = {}


def clean(text):
    text = NOISE_RE.sub(" ", SOURCE_RE.sub(" ", text)).lower()   # dấu hiệu nguồn tin trước, nhiễu sau
    text = re.sub(r"['’]s\b", "", text)                          # sở hữu cách: trump's -> trump
    return re.sub(r"[^a-z]+", " ", text)


def tokenize(text):
    return text.split()


def pos_tag(tokens):
    return tagger.tag(tokens)                     # [(từ, nhãn)], vd ('voted', 'VBD')


def remove_stopwords(tagged):
    return [(w, t) for w, t in tagged if w not in STOP_WORDS and len(w) > 1]


def lemma(word, tag):
    key = (word, POS_MAP.get(tag[0], "n"))
    if key not in _lemma_cache:                   # cache: mỗi cặp (từ, từ loại) chỉ lemmatize một lần
        _lemma_cache[key] = lemmatizer.lemmatize(*key)
    return _lemma_cache[key]


def lemmatize(tagged):
    return [lemma(w, t) for w, t in tagged]


def preprocess(text):
    """Một đoạn văn bản thô → chuỗi token đã làm sạch, cách nhau bởi dấu cách."""
    tokens = lemmatize(remove_stopwords(pos_tag(tokenize(clean(text)))))
    return " ".join(t for t in tokens if t not in STOP_WORDS and len(t) > 1)   # lọc lại: lemma có thể là stop-word ('was' -> 'be')


def preprocess_article(title, text):
    """Bài báo (tiêu đề, thân bài) → đầu vào của mô hình. Tiêu đề và thân bài xử lý riêng
    để regex dòng đề mục "WASHINGTON (Reuters) -" chỉ tác động lên đầu thân bài."""
    return f"{preprocess(title or '')} {preprocess(text or '')}".strip()


def _preprocess_batch(texts):
    return [preprocess(t) for t in texts]


def parallel_preprocess(series, n_chunks=64, n_jobs=-1):
    """Áp dụng `preprocess` cho cả một cột, chạy song song trên mọi nhân CPU (POS tagging khá chậm)."""
    chunks = np.array_split(series.fillna("").to_numpy(), n_chunks)
    out = Parallel(n_jobs=n_jobs)(delayed(_preprocess_batch)(c) for c in chunks)
    return pd.Series([x for c in out for x in c], index=series.index)
