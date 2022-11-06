import gunicorn
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

#Fonction qui permet l'extraction des tokens avec transformation en minuscule et suppression des ponctuations
def tokenize_lower_alpha_transform (corpus):
	tokens = word_tokenize(corpus.lower())
	lower_alpha_tokens = [ word for word in tokens if word.isalpha()]
	return lower_alpha_tokens

#Fonction qui permet de supprimer les stop words
nltk.download('stopwords')
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
		#Chargement du modéle
		loaded_model = pickle.load(open('LDA_model.sav', 'rb'))
		dataframe=pd.DataFrame()
		question = request.form.get('question')
		
		if question!="" :
			dataframe = pd.DataFrame({"question":[question]})
			bow_vector = dictionary.doc2bow(transform_bow_lem(title).split())
			for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
                if index = 0 :
                    prediction = "Score: {}\t Topic: {}".format(score, lda_model.print_topic(0, 5))	
		else :
			prediction = ""
	else :
		prediction = ""
	return render_template("index.html", output = prediction)

# Running the app
if __name__ == "__main__":
	import os from pml
	port = int(os.getenv('PORT'))
	app.run(host='0.0.0.0', port=port)