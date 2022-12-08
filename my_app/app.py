import gunicorn
import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask, request, render_template
from gensim.corpora import Dictionary
import pickle
import gensim
import nltk
from gensim.parsing.preprocessing  import preprocess_string

app = Flask(__name__)

#Fonction qui permet l'extraction des tokens avec transformation en minuscule et suppression des ponctuations
def tokenize_lower_alpha_transform (corpus):
	tokens = word_tokenize(corpus.lower())
	lower_alpha_tokens = [ word for word in tokens if word.isalpha()]
	return lower_alpha_tokens

#Fonction qui permet de supprimer les stop words
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_word = list(set(stopwords.words('english')))
def stop_word_filter(words) :
	no_stop_word = [word for word in words if not word in stop_word]
	return no_stop_word

# Fonction qui permet de lemmatizer (base d'un mot)
def lemmatize(words) :
	lemmatizer = WordNetLemmatizer()
	lematized = [lemmatizer.lemmatize(word) for word in words]
	return lematized

# Fonction de préparation du texte pour le bag of words et le Wors2Vec
def transform_bow_lem(words) :
	word_tokens = tokenize_lower_alpha_transform(words)
	stop_word = stop_word_filter(word_tokens)
	lemmatized_word = lemmatize(stop_word)    
	transf_desc_text = ' '.join(lemmatized_word)
	return transf_desc_text

@app.route('/', methods=['GET', 'POST'])
def index():
	#If a form is submitted
	if request.method == "POST" :
		prediction = ""
		question = ""
		#Chargement du modéle
		loaded_tfidf = pickle.load(open('tfidf.sav', 'rb'))
		question = request.form.get('question')
		
		if question!="" :
			title_transformed = transform_bow_lem(question).split()
			corpus = [title_transformed]
			tfidf = loaded_tfidf.transform(corpus)
			feature_array = np.array(loaded_tfidf.get_feature_names())
			tfidf_sorting = np.argsort(tfidf.toarray()).flatten()[::-1]
			n = 5
			top_n = feature_array[tfidf_sorting][:n]
			prediction = "Tags: {}".format(top_n[0] + " " + top_n[1] + " " + top_n[2] + " " + top_n[3] + " " + top_n[4])
		else :
			prediction = ""
	else :
		prediction = ""
	question = ""
	return render_template("index.html", output = prediction)

# Running the app
if __name__ == "__main__":
	port = int(os.getenv('PORT'))
	app.run(host='0.0.0.0', port=port)
