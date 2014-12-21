---
layout: default
img: eat-one-thing-.png
title: Word Cloud Recommender
category: Data Science
description: 


excerpt: I had the idea of building a recommender/predictor for what restaurants a yelp user might like using the reviews they have written, and building a word cloud according to a word neural network using contextual similarity and and unsupervised sentiment. 



I had the idea of building a recommender/predictor for what restaurants a yelp user might like using the reviews they have written, and building a word cloud according to a word neural network using contextual similarity and unsupervised sentiment.

**Algorithmic Methodology**

**1.** Filter the Yelp Academic Dataset to yield only food based venues

**2.** Using the [gensim](https://radimrehurek.com/gensim) library, I attempt to establish contextual similarity between words through sentence vector comparisons. The model looks at the context in each word is mentioned an establishes similarity based on the surrounding words. You can read about it greater detail on [word2vec](https://radimrehurek.com/2013/09/deep-learning-with-word2vec-and-gensim)  It is a great tool to build vectorization models using a new data set of text.

**3.** The next step is to do entity extraction on each review to find out what each user is talking about in the review itself. There is a great paper on this describing it [here](https://ov-research.uwaterloo.ca/papers/Vechtomova_JASIST2013.pdf). I chose a more cut and dry approach and used an established grammar for extracting noun phrases used [here](https://textblob.readthedocs.org/en/latest/_modules/textblob/en/np_extractors.html).

**4.** After extracting noun phrases, I sought to look for an unsupervised or weakly supervised sentiment algorithm that uses probabilistic phrases with regard to entities within each text block to determine sentiment. A fairly useful algorithm was developed by [TextBlob](https://textblob.readthedocs.org/en/latest/index.html). I plan on using this to identify which sentences the entities in step 2 were extracted from and doing a sentence level disambiguated subjective analysis will give me a better idea of how the user actually felt about the entity in question.

**5.** For the recommendation part, I plan on looking at the extracted entities of a user determine the sentiment associated with each enitity, and if the sentiment meets a certain threshold, then I will use the [word2vec](https://radimrehurek.com/2013/09/deep-learning-with-word2vec-and-gensim) model to determine other entities in the corpus with contextual similarity and recommend a small list restaurants that meet a certain sentiment requirement from other users. Then,  a list of venues will be compiled according to these featuresand then presented.

You must download the yelp academic dataset [here](https://www.yelp.com/academic_dataset)

If you have a badass machine that handle the computational needs of doing this, then use your RAM to work on this dataset, but I have 8gb RAM and I found it useful to work in a database such as MongoDB.

**settings.py** file to help with configurations
{% highlight python %}
Class Settings:
    def __init__(self):
        pass

    DATASET_FILE = '/home/specialk/dsf/DSFolder/yelp_academic_dataset_review.json'
    BUSINESS_FILE = '/home/specialk/dsf/DSFolder/yelp_academic_dataset_business.json'
    MONGO_CONNECTION_STRING = "mongodb://localhost:27030/"
    REVIEWS_DATABASE = "Dataset_Challenge_Reviews"
    TAGS_DATABASE = "Tags"
    REVIEWS_COLLECTION = "Reviews"
    MERGED_REVIEWS = "Reviews_Merged"
    CORPUS_COLLECTION = "Corpus"
{% endhighlight %}


Part **1** filters throught the yelp academic dataset and pulls only restaurants and food based venues
{% highlight python %}
import os
import time
import json

from pymongo import MongoClient

from settings import Settings



dataset_file = Settings.DATASET_FILE
business_file = Settings.BUSINESS_FILE
reviews_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.REVIEWS_DATABASE][
    Settings.REVIEWS_COLLECTION]

count = 0
done = 0
start = time.time()

with open(dataset_file) as dataset:
    count = sum(1 for line in dataset)

restaurant_dict = {}


with open(business_file) as businesses:
	next(businesses)
	for b in businesses:
		try:
			venue = json.loads(b)
		except ValueError:
			print 'Not Valid!!!!!'
		if 'Restaurants' in venue['categories']:
			restaurant_dict[venue['business_id']] = 1


with open(dataset_file) as dataset:
    next(dataset)
    for line in dataset:
        try:
            data = json.loads(line)
        except ValueError:
            print 'Oops!'
        if (data["type"] == "review") and (data["business_id"] in restaurant_dict):
            reviews_collection.insert({
                "reviewId": data["review_id"],
                "userId":  data["user_id"],
                "business": data["business_id"],
                "text": data["text"]
            })

        done += 1
        if done % 100 == 0:
            end = time.time()
            os.system('cls')
            print 'Done ' + str(done) + ' out of ' + str(count) + ' in ' + str((end - start))
{% endhighlight %}

Part **2** is where we build the neural network model using word2vec
{% highlight python %}
from pymongo import MongoClient
from settings import Settings
import gensim
from gensim.models import Word2Vec
from gensim.models import Phrases
import nltk
import os
import re, string
from unicodedata import category
from gensim.models.word2vec import logger, FAST_VERSION
import logging
logger.setLevel(logging.INFO)

def get_businesses():
    businesses = {}
    reviews_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.REVIEWS_DATABASE][
            Settings.REVIEWS_COLLECTION]
    reviews_cursor = reviews_collection.find()
    for review in reviews_cursor:
        if review["business"] not in businesses: businesses[review["business"]] = 1

    return businesses.keys()

def load_stopwords():
    stopwords = {}
    with open('stopwords.txt', 'rU') as f:
        for line in f:
            stopwords[line.strip()] = 1

    return stopwords

def notstop(word):
    if word is not None:
        if word not in load_stopwords():
            if word not in string.punctuation:
                return word
    return word

def sent_list():
    stopwords = load_stopwords()
    businesses = {}
    sentence_list = []
    c = 0
    reviews_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.REVIEWS_DATABASE][Settings.REVIEWS_COLLECTION]
    reviews_cursor = reviews_collection.find()
    for review in reviews_cursor:
        businesses[review['business']] = 1
        sentences = nltk.sent_tokenize(review["text"].lower())
        for sentence in sentences:
            word_list = []
            for word in nltk.word_tokenize(sentence):
                if word not in load_stopwords():
                    if word not in string.punctuation:
                        if word.isalpha():
                            word_list.append(word)
            print len(businesses.keys())
            if len(word_list) != 0:
                sentence_list.append(word_list)
        if len(businesses.keys()) == 5000:
            break
        with open('businesses.txt', 'w') as f:
            for business in businesses.keys():
                f.write(business+'\n')
    return sentence_list

def sent_Gen():
    stopwords = load_stopwords()
    businesses = {}
    
    c = 0
    reviews_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.REVIEWS_DATABASE][Settings.REVIEWS_COLLECTION]
    reviews_cursor = reviews_collection.find()
    for review in reviews_cursor:
        businesses[review['business']] = 1
        sentences = nltk.sent_tokenize(review["text"].lower())
        for sentence in sentences:
            word_list = []
            for word in nltk.word_tokenize(sentence):
                if word not in load_stopwords():
                    if word not in string.punctuation:
                        if word.isalpha():
                            word_list.append(word)
            print word_list
            if len(word_list) != 0 and len(businesses.keys()) <= 1:
                yield word_list

print 'Done building the sentence generator'

sentences = sent_list()
# sentences = sent_Gen()
skpg = 4
# phrasemodel = Phrases(sentences,min_count=10)
model = Word2Vec(sentences,min_count=10,size=50,workers=4,sg=skpg)
# os.system("touch models/yelp_phrase_model")
os.system("touch models/yelp_model"+str(skpg))
model.save('models/yelp_model'+str(skpg))
# phrasemodel.save('models/yelp_phrase_model')
{% endhighlight %}

Part **3** has two parts. The first is the phrase extraction algorithm using a context free grammar to identify noun phrases probabilistically, and the second is where we find out what each user is talking about in each review (entity extraction), the program extracts entities for each review and puts it into a datbase asynchronously

{% highlight python %}
# coding=UTF-8
import nltk
from nltk.corpus import brown
 
# This is a fast and simple noun phrase extractor (based on NLTK)
# Feel free to use it, just keep a link back to this post
# http://thetokenizer.com/2013/05/09/efficient-way-to-extract-the-main-topics-of-a-sentence/
# Create by Shlomi Babluki
# May, 2013
 
 
# This is our fast Part of Speech tagger
#############################################################################
brown_train = brown.tagged_sents(categories='reviews')
regexp_tagger = nltk.RegexpTagger(
    [(r'^-?[0-9]+(.[0-9]+)?$', 'CD'),
     (r'(-|:|;)$', ':'),
     (r'\'*$', 'MD'),
     (r'(The|the|A|a|An|an)$', 'AT'),
     (r'.*able$', 'JJ'),
     (r'^[A-Z].*$', 'NNP'),
     (r'.*ness$', 'NN'),
     (r'.*ly$', 'RB'),
     (r'.*s$', 'NNS'),
     (r'.*ing$', 'VBG'),
     (r'.*ed$', 'VBD'),
     (r'.*', 'NN')
])
unigram_tagger = nltk.UnigramTagger(brown_train, backoff=regexp_tagger)
bigram_tagger = nltk.BigramTagger(brown_train, backoff=unigram_tagger)
#############################################################################
 
 
# This is our semi-CFG; Extend it according to your own needs
#############################################################################
cfg = {}
cfg["NNP+NNP"] = "NNP"
cfg["NN+NN"] = "NNI"
cfg["NNI+NN"] = "NNI"
cfg["JJ+JJ"] = "JJ"
cfg["JJ+NN"] = "NNI"
#############################################################################
 
 
class NPExtractor(object):
 
    def __init__(self, sentence):
        self.sentence = sentence
 
    # Split the sentence into singlw words/tokens
    def tokenize_sentence(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        return tokens
 
    # Normalize brown corpus' tags ("NN", "NN-PL", "NNS" > "NN")
    def normalize_tags(self, tagged):
        n_tagged = []
        for t in tagged:
            if t[1] == "NP-TL" or t[1] == "NP":
                n_tagged.append((t[0], "NNP"))
                continue
            if t[1].endswith("-TL"):
                n_tagged.append((t[0], t[1][:-3]))
                continue
            if t[1].endswith("S"):
                n_tagged.append((t[0], t[1][:-1]))
                continue
            n_tagged.append((t[0], t[1]))
        return n_tagged
 
    # Extract the main topics from the sentence
    def extract(self):
 
        tokens = self.tokenize_sentence(self.sentence)
        tags = self.normalize_tags(bigram_tagger.tag(tokens))
 
        merge = True
        while merge:
            merge = False
            for x in range(0, len(tags) - 1):
                t1 = tags[x]
                t2 = tags[x + 1]
                key = "%s+%s" % (t1[1], t2[1])
                value = cfg.get(key, '')
                if value:
                    merge = True
                    tags.pop(x)
                    tags.pop(x)
                    match = "%s %s" % (t1[0], t2[0])
                    pos = value
                    tags.insert(x, (match, pos))
                    break
 
        matches = []
        for t in tags:
            if t[1] == "NNP" or t[1] == "NNI":
            #if t[1] == "NNP" or t[1] == "NNI" or t[1] == "NN":
                matches.append(t[0])
        return matches
 
 

{% endhighlight %}

{% highlight python %}
import multiprocessing
import time
import sys
import phrase_extraction
import nltk
from pymongo import MongoClient

from settings import Settings


def load_stopwords():
    stopwords = {}
    with open('stopwords.txt', 'rU') as f:
        for line in f:
            stopwords[line.strip()] = 1

    return stopwords


def worker(identifier, skip, count):
    done = 0
    start = time.time()

    stopwords = load_stopwords()
    reviews_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.REVIEWS_DATABASE][
        Settings.REVIEWS_COLLECTION]
    tags_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.TAGS_DATABASE][
        Settings.REVIEWS_COLLECTION]

    batch_size = 50
    for batch in range(0, count, batch_size):
        reviews_cursor = reviews_collection.find().skip(skip + batch).limit(batch_size)
        for review in reviews_cursor:

            np_extractor = phrase_extraction.NPExtractor(review["text"].lower())
        
            result = np_extractor.extract()

            if len(result) == 0:
                continue

            tags_collection.insert({
                "reviewId": review["reviewId"],
                "business": review["business"],
                "user_id": review["userId"],
                "text": review["text"],
                "entities": result 
            })

            done += 1
            if done % 100 == 0:
                end = time.time()
                print 'Worker' + str(identifier) + ': Done ' + str(done) + ' out of ' + str(count) + ' in ' + (
                    "%.2f" % (end - start)) + ' sec ~ ' + ("%.2f" % (done / (end - start))) + '/sec'
                sys.stdout.flush()


def main():
    reviews_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.REVIEWS_DATABASE][
        Settings.REVIEWS_COLLECTION]
    reviews_cursor = reviews_collection.find()
    count = reviews_cursor.count()
    workers = 8
    batch = count / workers

    jobs = []
    for i in range(workers):
        p = multiprocessing.Process(target=worker, args=((i + 1), i * batch, count / workers))
        jobs.append(p)
        p.start()

    for j in jobs:
        j.join()
        print '%s.exitcode = %s' % (j.name, j.exitcode)


if __name__ == '__main__':
    main()
{% endhighlight %}

Parts **4** and **5** are pretty much together as we gather the sentiment of each entity and based on the neural network we recommend venues according to the sentiment of a restaurant and how contextually similar their reviews are given by other users
{% highlight python %}
from pymongo import MongoClient
from settings import Settings
import gensim
from gensim.models import Word2Vec
from gensim.models import Phrases
import nltk
import os
import re, string
from unicodedata import category
from gensim.models.word2vec import logger, FAST_VERSION
import logging
logger.setLevel(logging.INFO)
from operator import itemgetter, attrgetter, methodcaller
from textblob import TextBlob
import numpy as np
import random


class ngram(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

# likelihood of recommendation
def strippunc(s):
    out = s.translate(string.maketrans("",""), string.punctuation)
    return out

def load_stopwords():
    stopwords = {}
    with open('stopwords.txt', 'rU') as f:
        for line in f:
            stopwords[line.strip()] = 1

    return stopwords

def notstop(word):
    if word is not None:
        if word not in load_stopwords():
            if word not in string.punctuation:
                return word
    return word


def sublistExists(list, sublist):
    for i in range(len(list)-len(sublist)+1):
        if sublist == list[i:i+len(sublist)]:
            return True #return position (i) if you wish
    return False

def intersection(input_list):
    return reduce(set.intersection,map(set,input_list)) 

def recommendation(userId,n):
    model = gensim.models.Word2Vec.load('models/yelp_model4')
    tags_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.TAGS_DATABASE][
        Settings.REVIEWS_COLLECTION]
    reviews_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.REVIEWS_DATABASE][Settings.REVIEWS_COLLECTION]
    users = ngram()
    ep = []
    most_sim = []
    restaurants = []
    
    rec_list= []
    max_sentiment = []
    wordcloud = []
    for rev in tags_collection.find({'user_id':userId}):
        for phrase in rev['entities']:
            if users[rev['user_id']]['ep'] == {}:
                users[rev['user_id']]['ep'] = [phrase]
            else:
                users[rev['user_id']]['ep'].append(phrase)
                users[rev['user_id']]['ep'] = list(set(users[rev['user_id']]['ep']))
        users[rev['user_id']]['restaurants'] = restaurants.append(rev['business'])
        
        zen = TextBlob(rev["text"])
        scores = [sentence.sentiment.polarity for sentence in zen.sentences]
        for j,sentence in enumerate([str(sentence) for sentence in zen.sentences]):
            for i,phrase in enumerate(users[rev['user_id']]['ep']):
                
                sentence = strippunc(sentence)
                
                if len(intersection([str(sentence).lower().split(),phrase.split()])) > 1 and np.mean(scores) > 0:
                    # print phrase
                    pinmodel = [(p in model.vocab) for p in phrase]
                    if len(pinmodel) == len(phrase):
                        max_sentiment.append((phrase,scores[j]))
        max_sentiment = sorted(max_sentiment, key=lambda maxsent: maxsent[1])
        # print max_sentiment
        for sent_tup in max_sentiment:
            # print sent_tup
            ls = sent_tup[0].split()
            for wd in ls:
                if wd not in model.vocab:
                    ls.remove(wd)

            lst = model.most_similar(ls,topn=20)
            rand_smpl = [ lst[i] for i in sorted(random.sample(xrange(len(lst)), 3)) ]
            most_sim.append(rand_smpl)
        # print most_sim    
        wordcloud = [item[0] for sublist in most_sim for item in sublist]
        wordcloud = set(wordcloud)
    # print wordcloud    
    for review in tags_collection.find():
        raw = TextBlob(review["text"])
        raw_scores = [sentence.sentiment.polarity for sentence in raw.sentences]
        # print np.mean(raw_scores)
        # print review["text"]
        if len(intersection([wordcloud,review["text"].lower().split()])) >= 5  and np.mean(raw_scores) > .2:
            rec_list.append((review['business'],review['text']))
            if len(rec_list) == n:
                return rec_list
    # find user entity phrases
    # find entity with highest sentiment associated with it and recommend based on that
    # get most similar words
    # find restaurants with highest summed contextual similarity using entity phrases, sort by this metric
    # then sort by sentiment perception using the full version of the review
    # recommend top n, make it user choice
    # maximum intersection in terms of similarity and then filter by sentiment
print recommendation("zvNimI98mrmhgNOOrzOiGg",10)

{% endhighlight %}

**Results comparison**

The following is an implementation of a collaborative filtering model and you can test the models to compare the results for research puprposes

{% highlight python %}
from scrapy.spider import BaseSpider
from scrapy.selector import HtmlXPathSelector
from scrapy.http import Request
import re

from pandoraFood.items import Review

# url string components for reviewer pages
URL_BASE = 'http://www.yelp.com/user_details_reviews_self?userid='
FILTER_SETTINGS = '&review_filter=category&category_filter=restaurants'

# yelp unique url endings for each restaurant
RESTAURANTS = ['z-and-y-restaurant-san-francisco', \
               'koi-palace-daly-city', \
               'ino-sushi-san-francisco', \
               'blackwood-san-francisco-3']

def createRestaurantPageLinks(self, response):
   reviewsPerPage = 40
   hxs = HtmlXPathSelector(response)
   totalReviews = int(hxs.select('//h2[@id="total_reviews"]/text()').extract()[0].strip().split(' ')[0])
   pages = [Request(url=response.url + '?start=' + str(reviewsPerPage*(n+1)), \
                    callback=self.parse) \
            for n in range(totalReviews/reviewsPerPage)]
   return pages

def createReviewerPageLinks(self, response):
   reviewsPerPage = 10
   hxs = HtmlXPathSelector(response)
   totalReviews = int(hxs.select('//div[@id="review_lister_header"]/em/text()').extract()[0].split(' ')[0])
   pages = [Request(url=response.url + '&rec_pagestart=' + str(reviewsPerPage*(n+1)), \
                    callback=self.parseReviewer) \
            for n in range(totalReviews/reviewsPerPage)]
   return pages

class RestaurantSpider(BaseSpider):
   name = 'crawlRestaurants'
   allowed_domains = ['yelp.com']
   start_urls = [ 'http://www.yelp.com/biz/%s' % s for s in RESTAURANTS]

   # default parse used for the landing page for each start_url
   def parse(self, response):
      requests = []

      # extract all reviews from the page and return a list of requests for the 5 star reviewers' profiles
      hxs = HtmlXPathSelector(response)
      userIDs = [userUrl.split('?userid=')[1] for \
                 userUrl in hxs.select('//li[@class="user-name"]/a/@href').extract()]
      ratings = hxs.select('//div[@id="reviews-other"]//meta[@itemprop="ratingValue"]/@content').extract()
   
      for i in range(len(ratings)):
         if float(ratings[i]) == 5:
            requests.append(Request(url=URL_BASE + userIDs[i] + FILTER_SETTINGS, \
                                    callback=self.parseReviewer))
   
      # request additional pages if we are on page 1 of the restaurant
      if response.url.find('?start=') == -1:
         requests += createRestaurantPageLinks(self, response)

      return requests
      
   # parse a given reviewer
   def parseReviewer(self, response):
      hxs = HtmlXPathSelector(response)
      restaurantUrls = hxs.select('//div[@class="review clearfix"]/ \
                                  div[@class="biz_info"]/h4/a/@href').extract()
      restaurants = [re.search(r'(?<=/biz/)[^#]*', rest).group() for rest in restaurantUrls]
      reviewerName = hxs.select('//title/text()').extract()[0].split('|')[0].replace('\'s Profile','').strip()
      reviewerUserID = re.search(r'(?<=userid=)[^&]*', response.url).group()
      ratingText = hxs.select('//div[@class="rating"]/i/@title').extract()
      ratings = [s.replace(' star rating','') for s in ratingText]

      reviews = []
      for i in range(len(restaurants)):
         review = Review()
         review['restaurant'] = restaurants[i]
         review['reviewerName'] = reviewerName
         review['reviewerUserID'] = reviewerUserID
         review['rating'] = float(ratings[i])
         reviews.append(review)

      # request additional pages if we are on page 1 of the reviewer
      additionalPages = []
      if response.url.find('&rec_pagestart=') == -1:
         additionalPages = createReviewerPageLinks(self, response)

      return reviews + additionalPages
{% endhighlight %}

**Concluding Thoughts**

I know what you are thinking! How do we know this pile of code is even good. The real answer is **We Don't**! This algorithm is almost totally unsupervised and truthfully impossible to test. One thing we do know is the results when read manually, do appear to be relevant and potential solutions to a recommendation problem based on the premise I have given and the style with which I chose to tackle this problem. Reviews for one thing are inherently biased and almost impossible to scientifically break down, so this will be a forever work in progress.


**Potential Improvements**

I want to come up with a review food corpus so that I can recognize food entities specifically and target only food entities to clean up the recommender.


**Editorial Notes/Me rambling about random stuff that you should probably maybe read** 

There are certain parameters in this program that can be changed to alter results. The results will vary everytime the algorithm is run but the selection criteria will remain the same. Permuting the selection will affect results, not necessarily a bad thing. It will be interesting to explore this. You can manipulate various thresholds such as the sentiment or similarity thresholds to tailor the algorithms criteria to your needs. For those wondering, the algorithm does yield coherent results, and the results will be further studied.
---
