import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import pandas as pd

import spacy
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

from num2words import num2words

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as Pipeline_imb

from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.matutils import cossim

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer


nltk.download('wordnet')
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

def cleanab(text):
    text = text.replace('-','')
    text = text.replace('African American','AfricanAmerican')
    return text



st.title('''Revising Sociology Abstracts''')
# st.markdown('''## From Paper to Manuscript''')
st.markdown('### A work in progress by [Neal Caren](https://nealcaren.org)')
st.write('As you turn your attention from your presenting your work at the ASAs to journal submission, you might focus on the differences between writing an abstract for a presentation versus writing one for publication.  This page might help.')
st.write('I analyzed five years of abstracts from the ASA annual meeting and sociology journals to see what words are disproportionately found in each. This page lets you use that model on your abstracts.')
st.write('To start, delete my abstract below, paste your abstract in the box, and press ⌘ + Enter to analyze your text:')
abstract = '''Can protest influence elections? We contribute to an important and emerging literature on the potential impact of social movements on electoral politics. We test the impact of protest during the Trump presidency on the 2020 election. We hypothesize that protests, both in opposition and in support of President Trump are likely to be associated with increasing voter participation and shape partisan voting patterns. To test these hypotheses, we use data collected from media accounts of protests, official election returns, combined with demographic and political control measures. Most important, we find strong evidence that anti-Trump protests are associated with county-level shifts in the percentage of Democratic voters. This effect is much greater in Republican counties, and these patterns suggest that protest helps to shift the composition of the electorate. We also find evidence of an increase in Democratic votes cast associated with anti-Trump protests in Republican counties, and a decrease in Republican votes cast associated with anti-Trump protests in Democratic counties. We find no evidence that pro-Trump protest increased Republican turnout or vote share. '''

sample_sentence = st.text_area('', abstract, height=400)



sample_sentence_length = word_counter(sample_sentence)
sample_sentence_slength = len(sent_tokenize(sample_sentence))
pasper = passive_per(sample_sentence) * 100
sample_sentence_slength_str = num2words(sample_sentence_slength)

sample_sentence = cleanab(sample_sentence)

st.markdown('### Descriptives')
st.write(f'Your abstract has {sample_sentence_length} words in {sample_sentence_slength_str} sentences. The median published abstract has 196 words in seven sentences.')
st.write(f'{pasper:.0f}% of your sentences were written in the passive voice. For published sociological research, the average is 20%.')

# Journal predictions
pub_est = load('pub_est.jlib')
def journal_choices(abstract):
    pred_prob = pub_est.predict_proba([abstract])
    pdf = pd.DataFrame(pred_prob, columns=pub_est.classes_)
    pdf = pdf.T.sort_values(by=0, ascending = False)

    journal_choices = pdf.index.values[:5]
    journal_choices = ', '.join(journal_choices[:-1]) + ' or ' + journal_choices[-1]
    return journal_choices

suggestions = journal_choices(sample_sentence)

st.markdown('### Journals')

st.write(f'You might think about submitting to journals like {suggestions}. Each publish work that have abstracts using similar words.')



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
used_df = used_df.sort_values(by=0)
negative_words = used_df[used_df[0] < -.5].index
negative_words = [w for w in negative_words if ' ' not in w][:10]

custom_syns = []
syn_dict = load('synonyms.jlib')
for word in negative_words:
	if word in syn_dict:
		custom_syns.append({'word': word, 'Possible replacements' :syn_dict[word] })
sdf = pd.DataFrame(custom_syns)
st.markdown('### Revisions')

st.write('When looking at the words you used, there might be some similar words that are more associated with journal abstracts than conference abstracts. Below is a chart that shows, for each of your words in red, the three  similar words not associated with conference abstracts. Since similarity is based on word embeddings trained on a pretty small corpus or because you might not want to mimic existing  work, specific suggestions might not actually be helpful.')
st.table(sdf.set_index('word'))

st.markdown('### Similar Articles')

nltk.download('stopwords')
from nltk.corpus import stopwords


def title_fix(title):
    if 'SOCIOLOGICAL METHODOLOGY' in title:
        title = 'SOCIOLOGICAL METHODOLOGY'
    title = title.title()
    for hack in [' Of ', ' And ', ' On ']:
        title = title.replace(hack, hack.lower())
    return title



def valid(token):
    if token.isnumeric() == True:
        return False
    if len(token) < 2:
        return False
    if token in stop_words:
        return False
    return True

def fast_tokenize(doc):
    doc = doc.lower()
    tokens = tokenizer.tokenize(doc)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if valid(t) == True]
    return tokens

def nearestneighbor(existing_tokens, ab_tokens):
    doc1 = topic_model[ab_tokens]
    doc2 = existing_tokens
    return cossim(doc1, doc2)

def simwork(new_abstract):
    ab_tokens = fast_tokenize(new_abstract)
    ab_tokens = topic_dictionary.doc2bow(ab_tokens)
    cdf['sim'] = cdf['topics'].apply(nearestneighbor, args=(ab_tokens,))
    matches = list(cdf.sort_values(by='sim', ascending=False)['cite'].values[:5])
    match_text = '\n* '.join(matches)
    match_text = '* ' + match_text.replace('..','.')
    return match_text

cdf = pd.read_json('pub_cites.json')
topic_model = LdaModel.load("abstract_topics.model")
topic_dictionary =  Dictionary.load("abstract_topics.dict")

lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')
stop_words = stopwords.words('english')

matches = simwork(sample_sentence)
st.write('You might have already seen these, but some recent work that might be similar to yours is:')
st.markdown(matches)

st.markdown('### Small print')
st.markdown('''This analysis is built on:

*Data:*
* 14,609 ASA conference abstracts presented from 2016-2019 collected from the ASA [conference paper archive](https://convention2.allacademic.com/one/asa/asa/).
* 6,590 articles published between 2017 and 2021 in sociology journals from Web of Science.

*Analysis:*
* Conference/publication classification is based on logistic regression model after performing a grid search to find the best term frequency vectorizer parameters.
* Word colors are based on logistic regression coefficients.
* The revisions options use word embeddings trained on a sample of papers submitted to the ASA to find words without negative coefficients that are found in a similar context.
* Article recommendations are based on cosine similarity of topic models estimates. Anything faster would be great.

*Software:*
* Coding in Python using Jupyter Notebooks.
* [scikit-learn](https://scikit-learn.org/stable/) for classification model.
* [spaCy](https://spacy.io) for estimating active/passive voice.
* [ELI5](https://eli5.readthedocs.io/en/latest/) for displaying word-specific predictions.
* [Gensim](https://radimrehurek.com/gensim/) for word embeddings and topic models.
* [streamlit](https://streamlit.io) for interactivity.
''')
