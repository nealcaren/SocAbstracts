import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import pandas as pd

import spacy

from spacy.cli import download
try:
    nlp = spacy.load('en_core_web_md')
except OSError:

    from spacy.cli import download
    download('en_core_web_md')
    nlp = spacy.load('en_core_web_md')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from string import punctuation

import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize, sent_tokenize


import eli5

from joblib import load

def word_counter(text):
	words = word_tokenize(text)
	words = [w for w in words if w not in punctuation]
	return len(words)

def passive_per(abstract):
    sentences = sent_tokenize(abstract)
    sen_count = len(sentences)
    pas_count = 0
    for sentence in sentences:
        doc = nlp(sentence)
        passive = False
        for t in doc:
            if t.dep_ in ['nsubjpass','auxpass']:
                passive = True
        if passive == True:
            pas_count +=1

    return(pas_count/sen_count)

st.title('From Paper to Manuscript')
st.markdown('### A work in progress by Neal Caren')
st.write('As you turn your attention from your presenting your work at the ASAs to journal submission, you might focus on the differences between writing an abstract for a presentation versus writing one for publication.  This page might help.')
st.write('I analyzed five years of abstracts from the ASA annual meeting and sociology journals to see what words are disproportionately found in each. This page lets you use that model on your abstracts.')
st.write('To start, delete my abstract below, paste your abstract in the box, and press ⌘ + Enter to analyze your text:')
abstract = '''Can protest influence elections? We contribute to an important and emerging literature on the potential impact of social movements on electoral politics. We test the impact of protest during the Trump presidency on the 2020 election. We hypothesize that protests, both in opposition and in support of President Trump are likely to be associated with increasing voter participation and shape partisan voting patterns. To test these hypotheses, we use data collected from media accounts of protests, official election returns, combined with demographic and political control measures. Most important, we find strong evidence that anti-Trump protests are associated with county-level shifts in the percentage of Democratic voters. This effect is much greater in Republican counties, and these patterns suggest that protest helps to shift the composition of the electorate. We also find evidence of an increase in Democratic votes cast associated with anti-Trump protests in Republican counties, and a decrease in Republican votes cast associated with anti-Trump protests in Democratic counties. We find no evidence that pro-Trump protest increased Republican turnout or vote share. '''

sample_sentence = st.text_area('', abstract, height=400)



sample_sentence_length = word_counter(sample_sentence)
sample_sentence_slength = len(sent_tokenize(sample_sentence))
pasper = passive_per(sample_sentence) * 100

st.markdown('### Descriptives')
st.write(f'Your abstract has {sample_sentence_length} words in {sample_sentence_slength} sentences. The median published abstract has 196 words in seven sentences.')
st.write(f'{pasper:.0f}% of your sentences were written in the passive voice. For published sociological research, the average is 20%.')

# Prediction from saved model
pipeline = load('abstract_estimator.joblib')
vectorizer = pipeline.named_steps['vect']
model = pipeline.named_steps['clf']

pred = pipeline.predict_proba([sample_sentence])[0][-1]

if pred > .5:
	ptext = 'found in a journal than in a conference.'
else:
	ptext = 'found in a conference than a journal'

pstatement = f'Based on my data and model, the probablity that this would associated with a journal is {pred:.2f}.'

st.markdown('### Words')

st.write(f'Looking at the words you used, compared to other abstracts from sociology journals and prior year ASAs, your abstract looks more like one {ptext} {pstatement}')
st.write('Here is a summary of what words in your abstract the model found associated with journal and conference abstracts. Words in pink and red are those more likely to be found in conference abstracts, while those in green are disproporatenly found in the abstracts of published papers.')
def eli5_abstract():
    sample_pred = eli5.show_prediction(model,
                         sample_sentence,
						 target_names = {True: 'Publication', False: 'Presentation'},
						 targets = [True],
                         vec = vectorizer).data
    sample_pred = sample_pred.replace('\n',' ')
	#exclude the table at the top
    sample_pred = sample_pred.split('</table>')[-1]
    return sample_pred
st.markdown(eli5_abstract(), unsafe_allow_html=True)


# Find useful synonyms
coef_df = pd.DataFrame(model.coef_.T,
                      index = vectorizer.get_feature_names())
coef_df['used'] = vectorizer.transform([sample_sentence]).toarray()[0]
used_df = coef_df[coef_df['used']!=0]
negative_words = used_df[used_df[0] < 0].index
negative_words = [w for w in negative_words if ' ' not in w]

custom_syns = []
syn_dict = load('synonyms.jlib')
for word in negative_words:
	if word in syn_dict:
		custom_syns.append({'word': word, 'Possible replacements' :syn_dict[word] })
sdf = pd.DataFrame(custom_syns)
st.markdown('### Revisions')

st.write('When looking at the words you used, there might be some similar words that are more associated with journal abstracts than conference abstracts. Below is a chart that shows, for each your words in red, the three most similar words positively associated with journal abstracts. Since similarity is based on word embeddings, specific suggestions might not actually be useful.')
st.table(sdf.set_index('word'))

st.markdown('### Small print')
st.markdown('''This analysis is built on:

*Data:*
* [x] ASA conference abstracts from the ASA website from [x] to [y].
* [y] articles published in sociology journals from Web of Science from [x] to [y].

*Analysis:*
* Conference/publication classification is based on logistic regression model after performing a grid search to find the best term frequency vectorizer paramaters.
* Word colors are based on logistic regression coefficients.
* The revisions options use word embeddings trained on the entire abstract corpus to find words with positive coefficients that are found in a similar context.

*Software:*
* Coding in Python using Jupyter Notebooks.
* [scikit-learn](https://scikit-learn.org/stable/) for classification model.
* [spaCy](https://spacy.io) for estimating active/passive voice.
* [ELI5](https://eli5.readthedocs.io/en/latest/) for displaying word-specific predictions.
* [Gensim](https://radimrehurek.com/gensim/) for word embeddings.
* [streamlit](https://streamlit.io) for interactivity.
''')
