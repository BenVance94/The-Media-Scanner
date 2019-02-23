print("Clear Eyes. Full Heart. Can't Lose.")
import pandas as pd
import requests
import datetime
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
year = datetime.datetime.today().strftime('%Y')
month = datetime.datetime.today().strftime('%m')
day = datetime.datetime.today().strftime('%d')
directory = 'Data_Archive/Year={0}/Month={1}/Day={2}/'.format(year,month,day)
if os.path.exists(directory):
	directory = 'Data_Archive_Extras/Year={0}/Month={1}/Day={2}/'.format(year,month,day)
	print("Re-Directing Data to Data Archive Extras")
else:
	print("Data will be directed to " + str(directory))
	os.makedirs(directory)
	
df_america = pd.DataFrame(columns=['headline_descriptions','news_source','coordinating_conjunction','cardinal_digit', 'determiner', 'existential', 'foreign', 'preposition', 'adjective','numbering', 'modal', 'noun', 'possessive', 'pronoun', 'adverb','giveup', 'to_go', 'interjection', 'verb'])

media_list = ['wsj','cnbc','fortune','businessinsider']

for media_code in media_list:
	url = 'https://www.' + str(media_code) + '.com/'
	r = requests.get(url)
	if r.status_code == 200:
		print(str(media_code) + " was sucessfully scanned")
	else:
		print("There was an error with " + str(media_code) + " || Code: " + str(r.status_code))
	soup = BeautifulSoup(r.content, 'html.parser')
	textContent = []
	num0=1000
	for i in range(0, num0):
		try:
			if media_code == 'wsj':
				paragraphs = soup.find_all("p")[i].text

			elif media_code == 'cnbc' or media_code == 'fortune' or media_code == 'businessinsider':
				paragraphs = soup.find_all("a")[i].text
				paragraphs = paragraphs.replace("\n", "")
				paragraphs = paragraphs.strip()
				
			if len(str(paragraphs)) > 20:
				textContent.append(paragraphs)
				
		except IndexError:
			num0 = len(textContent)
	print(str(media_code) + " has about " + str(num0) + " descriptions - COMPLETED")
	df = pd.DataFrame(textContent, columns=['headline_descriptions'])
	df['news_source'] = media_code
	
	def vader_anal(row):
		sentence = str(row)
		tokenized_sentence = nltk.word_tokenize(sentence)

		sid = SentimentIntensityAnalyzer()
		pos_word_list=[]
		neu_word_list=[]
		neg_word_list=[]

		for word in tokenized_sentence:
			if (sid.polarity_scores(word)['compound']) >= 0.1:
				pos_word_list.append(word)
			elif (sid.polarity_scores(word)['compound']) <= -0.1:
				neg_word_list.append(word)
			else:
				neu_word_list.append(word) 
		score = sid.polarity_scores(sentence)
		return pd.Series((score['compound'],score['pos'],score['neu'],score['neg'],pos_word_list,neu_word_list,neg_word_list))
	
	def part_of_speech(row):
		pos_type_list=[]
		coordinating_conjunction=0
		cardinal_digit=0
		determiner=0
		existential=0
		foreign=0
		preposition=0
		adjective=0
		numbering=0
		modal=0
		noun=0
		possessive=0
		pronoun=0
		adverb=0
		giveup=0
		to_go=0
		interjection=0
		verb=0
		tokens=nltk.word_tokenize(str(row))
		for row0 in nltk.pos_tag(tokens):
			pos_type_list.append(row0[1])
		for row1 in pos_type_list:
			if 'CC' in row1:
				coordinating_conjunction = coordinating_conjunction + 1
			elif 'CD' in row1:
				cardinal_digit = cardinal_digit + 1
			elif 'DT' in row1:
				determiner = determiner + 1
			elif 'EX' in row1:
				existential = existential + 1
			elif 'FW' in row1:
				foreign = foreign + 1
			elif 'IN' in row1:
				preposition = preposition + 1
			elif 'JJ' in row1:
				adjective = adjective + 1
			elif 'LS' in row1:
				numbering = numbering + 1
			elif 'MD' in row1:
				modal = modal + 1
			elif 'NN' in row1:
				noun = noun + 1
			elif 'POS' in row1:
				possessive = possessive + 1
			elif 'PRP' in row1 or 'WP' in row1:
				pronoun = pronoun + 1
			elif 'RB' in row1:
				adverb = adverb + 1
			elif 'RP' in row1:
				giveup = giveup + 1
			elif 'TO' in row1:
				to_go = to_go + 1
			elif 'UH' in row1:
				interjection = interjection + 1
			elif 'VB' in row1:
				verb = verb + 1
		return pd.Series((coordinating_conjunction,cardinal_digit,determiner,existential,foreign,preposition,adjective,numbering,modal,noun,possessive,pronoun,adverb,giveup,to_go,interjection,verb))
	
	
	df[['score_coumpound','score_pos','score_neu','score_neg','pos_words','neu_words','neg_words']] = df['headline_descriptions'].apply(vader_anal)
	df[['coordinating_conjunction','cardinal_digit','determiner','existential','foreign','preposition','adjective','numbering','modal','noun','possessive','pronoun','adverb','giveup','to_go','interjection','verb']] = df['headline_descriptions'].apply(part_of_speech)
	
	df_america = pd.concat([df_america,df],ignore_index=True, sort=True)
	
	
	#filename1 = str(directory) + "{0}_descriptions_{1}.csv".format(media_code,datetime.datetime.today().strftime('%Y-%m-%d'))
	filename2 = str(directory) + "{0}_html_{1}".format(media_code,datetime.datetime.today().strftime('%Y-%m-%d'))

	#df.to_csv(filename1)
	#print(str(filename1) + " has been created")
	f = open('{}.txt'.format(filename2),'w')
	f.write(str(r.content))
	f.close()
	
filename0 = str(directory) + "media_descriptions_{}.csv".format(datetime.datetime.today().strftime('%Y-%m-%d'))	
df_america.to_csv(filename0)
df_verify = pd.read_csv(filename0)
print("Data Movement Check: " + str(len(df_verify) == len(df_america)))