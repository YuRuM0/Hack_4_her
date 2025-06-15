import genderdecoder3
import re
import pandas as pd
from collections import Counter
from genderdecoder3 import assess
from sklearn.feature_extraction.text import TfidfVectorizer
from deep_translator import GoogleTranslator
import re

df = pd.read_csv("labelled_cleaned.csv")
df.head()

feminine_coded_stems_en = [
	"agree", "affectionate", "child", "cheer", "collab", "commit", "communal",
	"compassion", "connect", "considerate", "cooperat", "co-operat", "depend",
	"emotiona", "empath", "feel", "flatterable", "gentle", "honest",
	"interpersonal", "interdependen", "interpersona", "inter-personal", "inter-dependen",
	"inter-persona", "kind", "kinship", "loyal", "modesty", "nag", "nurtur",
	"pleasant", "polite", "quiet", "respon", "sensitiv", "submissive", "support",
	"sympath", "tender", "together", "trust", "understand", "warm", "whin",
	"enthusias", "inclusive", "yield", "share", "sharin"
]
masculine_coded_stems_en = [
	"active", "adventurous", "aggress", "ambitio", "analy", "assert", "athlet",
	"autonom", "battle", "boast", "challeng", "champion", "compet", "confident",
	"courag", "decid", "decision", "decisive", "defend", "determin", "domina",
	"dominant", "driven", "fearless", "fight", "force", "greedy", "head-strong",
	"headstrong", "hierarch", "hostil", "impulsive", "independen", "individual",
	"intellect", "lead", "logic", "objective", "opinion", "outspoken", "persist",
	"principle", "reckless", "self-confiden", "self-relian", "self-sufficien",
	"selfconfiden", "selfrelian", "selfsufficien", "stubborn", "superior", "unreasonab"
]

def translate_stems(stems, source="en", target="nl"):
	translator = GoogleTranslator(source=source, target=target)
	translations = []
	for stem in stems:
		try:
			translations.append(translator.translate(stem))
		except Exception:
			translations.append(stem)
	return translations

# Translated
feminine_coded_stems_nl = translate_stems(feminine_coded_stems_en)
masculine_coded_stems_nl = translate_stems(masculine_coded_stems_en)

# Combined
feminine_coded_stems = feminine_coded_stems_en + feminine_coded_stems_nl
masculine_coded_stems = masculine_coded_stems_en + masculine_coded_stems_nl

def count_partial_matches(tokens, stem_list):
	return sum(any(stem in token for stem in stem_list) for token in tokens)

def analyze_bias(df, text_column="job_description"):
	fem_counts, masc_counts, bias_labels, bias_scores = [], [], [], []

	for _, row in df.iterrows():
		text = str(row[text_column])
		tokens = re.findall(r'\b\w+\b', text.lower())
		fem_count = count_partial_matches(tokens, feminine_coded_stems)
		masc_count = count_partial_matches(tokens, masculine_coded_stems)
		bias_score = masc_count - fem_count

		if bias_score > 0:
			label = "Masculine biased"
		elif bias_score < 0:
			label = "Feminine biased"
		else:
			label = "Neutral"

		fem_counts.append(fem_count)
		masc_counts.append(masc_count)
		bias_scores.append(bias_score)
		bias_labels.append(label)

	df["feminine_word_count"] = fem_counts
	df["masculine_word_count"] = masc_counts
	df["bias_score"] = bias_scores
	df["bias_label"] = bias_labels
	return df

def tfidf_bias_score(tfidf_vector, feature_names, masc_stems, fem_stems):
	tfidf_scores = dict(zip(feature_names, tfidf_vector.toarray()[0]))
	masc_score = sum(score for word, score in tfidf_scores.items() if any(stem in word for stem in masc_stems))
	fem_score = sum(score for word, score in tfidf_scores.items() if any(stem in word for stem in fem_stems))
	return masc_score - fem_score, masc_score, fem_score

df = analyze_bias(df)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['job_description'].astype(str))
feature_names = vectorizer.get_feature_names_out()

tfidf_bias, tfidf_masc, tfidf_fem = [], [], []
for i in range(tfidf_matrix.shape[0]):
	bias, masc, fem = tfidf_bias_score(tfidf_matrix[i], feature_names, masculine_coded_stems, feminine_coded_stems)
	tfidf_bias.append(bias)
	tfidf_masc.append(masc)
	tfidf_fem.append(fem)

df['tfidf_bias_score'] = tfidf_bias
df['tfidf_masculine_score'] = tfidf_masc
df['tfidf_feminine_score'] = tfidf_fem

def tfidf_bias_score_sparse(tfidf_vector, feature_names, masc_stems, fem_stems):
	# Only check words that actually appear in the current job ad
	indices = tfidf_vector.nonzero()[1]
	masc_score = 0
	fem_score = 0
	for idx in indices:
		word = feature_names[idx]
		score = tfidf_vector[0, idx]
		if any(stem in word for stem in masc_stems):
			masc_score += score
		if any(stem in word for stem in fem_stems):
			fem_score += score
	return masc_score - fem_score, masc_score, fem_score

# Now run it!
tfidf_bias, tfidf_masc, tfidf_fem = [], [], []
for i in range(tfidf_matrix.shape[0]):
	bias, masc, fem = tfidf_bias_score_sparse(
		tfidf_matrix[i], feature_names, masculine_coded_stems, feminine_coded_stems)
	tfidf_bias.append(bias)
	tfidf_masc.append(masc)
	tfidf_fem.append(fem)

df['tfidf_bias_score'] = tfidf_bias
df['tfidf_masculine_score'] = tfidf_masc
df['tfidf_feminine_score'] = tfidf_fem

job_descriptions = df['job_description'].astype(str).tolist()
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(job_descriptions)
feature_names = vectorizer.get_feature_names_out()

def tfidf_bias_score(tfidf_vector, feature_names, masc_words, fem_words):
	tfidf_scores = dict(zip(feature_names, tfidf_vector.toarray()[0]))
	masc_score = sum(tfidf_scores.get(word, 0) for word in masc_words)
	fem_score = sum(tfidf_scores.get(word, 0) for word in fem_words)
	return masc_score - fem_score

# Apply decoding and scoring per job
bias_scores = []
gender_labels = []
masc_words = []
fem_words = []

for i, text in enumerate(job_descriptions):
	result = assess(text)  # returns a dict directly
	masc_words += result['masculine_coded_words']
	fem_words += result['feminine_coded_words']
	bias_score = tfidf_bias_score(tfidf_matrix[i], feature_names, masc_words, fem_words)

	bias_scores.append(bias_score)
	gender_labels.append(result['result'])

def classify_coding(masc_words, fem_words):
	if len(masc_words) > len(fem_words):
		return "masculine-coded", (len(masc_words) - len(fem_words))
	elif len(fem_words) > len(masc_words):
		return "feminine-coded", (len(masc_words) - len(fem_words))
	else:
		return "neutral", (len(masc_words) - len(fem_words))
