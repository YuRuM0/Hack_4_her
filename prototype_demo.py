import sys
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

def tfidf_bias_score(tfidf_vector, feature_names, masc_words, fem_words):
	tfidf_scores = dict(zip(feature_names, tfidf_vector.toarray()[0]))
	masc_score = sum(tfidf_scores.get(word, 0) for word in masc_words)
	fem_score = sum(tfidf_scores.get(word, 0) for word in fem_words)
	return masc_score - fem_score

def collect_partial_matches(tokens, stem_list):
	return [token for token in tokens for stem in stem_list if stem in token]

def classify_coding(masc_words, fem_words):
	if len(masc_words) > len(fem_words):
		return "masculine-coded"
	elif len(fem_words) > len(masc_words):
		return "feminine-coded"
	else:
		return "neutral"

feminine_coded_words = [
	"agree", "affectionate", "child", "cheer", "collab", "commit", "communal", "compassion",
	"connect", "considerate", "cooperat", "co-operat", "depend", "emotiona", "empath", "feel",
	"flatterable", "gentle", "honest", "interpersonal", "interdependen", "interpersona", "inter-personal",
	"inter-dependen", "inter-persona", "kind", "kinship", "loyal", "modesty", "nag", "nurtur",
	"pleasant", "polite", "quiet", "respon", "sensitiv", "submissive", "support", "sympath",
	"tender", "together", "trust", "understand", "warm", "whin", "enthusias", "inclusive",
	"yield", "share", "sharin", "mee eens zijn", "geliefd", "kind", "aanmoedigen", "samenwerken",
	"verbinden", "gemeenschappelijk", "medeleven", "verbinden", "attent", "Cooperat", "coöperat",
	"afhankelijk zijn", "Emota", "empathie", "gevoel", "platterbaar", "teder", "eerlijk",
	"interpersoonlijk", "onderlinge afhankelijke", "interpersona", "interpersoonlijk", "tussenaf",
	"inter-Persona", "vriendelijk", "verwantschap", "loyaal", "bescheidenheid", "zeuren", "koesteren",
	"prettig", "beleefd", "rustig", "respon", "gevoelig", "onderdanig", "steun", "sympathie",
	"teder", "samen", "vertrouwen", "begrijpen", "warm", "jochie", "enthousiasme", "inclusief",
	"opbrengst", "deel", "Sharin"
]

masculine_coded_words = [
	"active", "adventurous", "aggress", "ambitio", "analy", "assert", "athlet", "autonom", "battle", "boast", "challeng",
	"champion", "compet", "confident", "courag", "decid", "decision", "decisive", "defend", "determin", "domina", "dominant",
	"driven", "fearless", "fight", "force", "greedy", "head-strong", "headstrong", "hierarch", "hostil", "impulsive",
	"independen", "individual", "intellect", "lead", "logic", "objective", "opinion", "outspoken", "persist", "principle",
	"reckless", "self-confiden", "self-relian", "self-sufficien", "selfconfiden", "selfrelian", "selfsufficien",
	"stubborn", "superior", "unreasonab", "actief", "avontuurlijk", "agresseren", "ambitio", "analyseren", "beweren",
	"atlet", "autonom", "strijd", "opscheppen", "uitdagen", "kampioen", "concurreren", "vol vertrouwen", "courag",
	"beslissen", "beslissing", "besluitvol", "verdedigen", "bepalend", "domina", "dominant", "aangedreven",
	"onbevreesd", "gevecht", "kracht", "hebberig", "heagend", "eigenwijsheid", "hiërarch", "hospeld", "impulsief",
	"onafhankelijk", "individu", "intellect", "leiding", "logica", "objectief", "mening", "uitgesproken", "volharden",
	"beginsel", "roekeloos", "zelfverzekerd", "zelfrandelijk", "zelfvoorziening", "zelfconfideren", "zelfrelian",
	"zelfverzekerde", "koppig", "superieur", "onredelijk"
]

def main(dataframe):
	job_descriptions = dataframe['job_description'].astype(str).tolist()
	vectorizer = TfidfVectorizer()
	tfidf_matrix = vectorizer.fit_transform(job_descriptions)
	feature_names = vectorizer.get_feature_names_out()

	dataframe['feminine_words'] = None
	dataframe['masculine_words'] = None
	dataframe['feminine_word_count'] = 0
	dataframe['masculine_word_count'] = 0
	dataframe['score'] = 0.0
	dataframe['gender_label'] = None

	for idx, row in dataframe.iterrows():
		job_description_nl = str(row["job_description"])
		tokens = re.findall(r'\b\w+\b', job_description_nl.lower())

		fem_words = collect_partial_matches(tokens, feminine_coded_words)
		masc_words = collect_partial_matches(tokens, masculine_coded_words)
		bias_score = tfidf_bias_score(tfidf_matrix[idx], feature_names, masc_words, fem_words)

		dataframe.at[idx, "feminine_word_count"] = len(fem_words)
		dataframe.at[idx, "masculine_word_count"] = len(masc_words)
		dataframe.at[idx, "feminine_words"] = fem_words
		dataframe.at[idx, "masculine_words"] = masc_words
		dataframe.at[idx, "score"] = bias_score
		dataframe.at[idx, "gender_label"] = classify_coding(masc_words, fem_words)

	dataframe["masculine_words"] = dataframe["masculine_words"].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
	dataframe["feminine_words"] = dataframe["feminine_words"].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))

	label_map = {
		'masculine-coded': 1,
		'feminine-coded': -1,
		'neutral': 0
	}
	dataframe['gender_label_num'] = dataframe['gender_label'].map(label_map)
	dataframe['final_bias_score'] = 0.5 * dataframe['score'] + 0.5 * dataframe['gender_label_num']
	
	return dataframe

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Usage: python gender_bias_analysis.py <job_descriptions.txt>")
		sys.exit(1)

	input_file = sys.argv[1]

	with open(input_file, "r", encoding="utf-8") as f:
		text = f.read()
	job_descriptions = text.strip().split("\n\n")
	print(f"Total job descriptions found: {len(job_descriptions)}")

	df = pd.DataFrame({"job_description": job_descriptions})

	processed_df = main(df)

	for idx, row in processed_df.iterrows():
		print(f"Job Description #{idx+1}:\n")
		print(f"  Final Bias Score: {row['final_bias_score']:.4f}\n")
		print(f"  Masculine Words: {row['masculine_words']}\n")
		print(f"  Feminine Words: {row['feminine_words']}")
		print("-" * 40)
