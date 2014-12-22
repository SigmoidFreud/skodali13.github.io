---
layout: post
title: Word Cloud Recommender
img: eat-one-thing-.png
category: Data Science
description: <p><strong>Algorithmic Methodology</strong></p>

<p><strong>1.</strong> Filter the Yelp Academic Dataset to yield only food based venues</p>

<p><strong>2.</strong> Using the <a href="https://radimrehurek.com/gensim">gensim</a> library, I attempt to establish contextual similarity between words through sentence vector comparisons. The model looks at the context in each word is mentioned an establishes similarity based on the surrounding words. You can read about it greater detail on <a href="https://radimrehurek.com/2013/09/deep-learning-with-word2vec-and-gensim">word2vec</a>  It is a great tool to build vectorization models using a new data set of text.</p>

<p><strong>3.</strong> The next step is to do entity extraction on each review to find out what each user is talking about in the review itself. There is a great paper on this describing it <a href="https://ov-research.uwaterloo.ca/papers/Vechtomova_JASIST2013.pdf">here</a>. I chose a more cut and dry approach and used an established grammar for extracting noun phrases used <a href="https://textblob.readthedocs.org/en/latest/_modules/textblob/en/np_extractors.html">here</a>.</p>

<p><strong>4.</strong> After extracting noun phrases, I sought to look for an unsupervised or weakly supervised sentiment algorithm that uses probabilistic phrases with regard to entities within each text block to determine sentiment. A fairly useful algorithm was developed by <a href="https://textblob.readthedocs.org/en/latest/index.html">TextBlob</a>. I plan on using this to identify which sentences the entities in step 2 were extracted from and doing a sentence level disambiguated subjective analysis will give me a better idea of how the user actually felt about the entity in question.</p>

<p><strong>5.</strong> For the recommendation part, I plan on looking at the extracted entities of a user determine the sentiment associated with each enitity, and if the sentiment meets a certain threshold, then I will use the <a href="https://radimrehurek.com/2013/09/deep-learning-with-word2vec-and-gensim">word2vec</a> model to determine other entities in the corpus with contextual similarity and recommend a small list restaurants that meet a certain sentiment requirement from other users. Then,  a list of venues will be compiled according to these featuresand then presented.</p>

<p>You must download the yelp academic dataset <a href="https://www.yelp.com/academic_dataset">here</a></p>

<p>If you have a badass machine that handle the computational needs of doing this, then use your RAM to work on this dataset, but I have 8gb RAM and I found it useful to work in a database such as MongoDB.</p>

<p><strong>settings.py</strong> file to help with configurations
{% highlight python %}
Class Settings:
    def <strong>init</strong>(self):
        pass</p>

<pre><code>DATASET_FILE = '/home/specialk/dsf/DSFolder/yelp_academic_dataset_review.json'
BUSINESS_FILE = '/home/specialk/dsf/DSFolder/yelp_academic_dataset_business.json'
MONGO_CONNECTION_STRING = "mongodb://localhost:27030/"
REVIEWS_DATABASE = "Dataset_Challenge_Reviews"
TAGS_DATABASE = "Tags"
REVIEWS_COLLECTION = "Reviews"
MERGED_REVIEWS = "Reviews_Merged"
CORPUS_COLLECTION = "Corpus"
</code></pre>

<p>{% endhighlight %}</p>

<p>Part <strong>1</strong> filters throught the yelp academic dataset and pulls only restaurants and food based venues
{% highlight python %}
import os
import time
import json</p>

<p>from pymongo import MongoClient</p>

<p>from settings import Settings</p>

<p>dataset<em>file = Settings.DATASET</em>FILE
business<em>file = Settings.BUSINESS</em>FILE
reviews<em>collection = MongoClient(Settings.MONGO</em>CONNECTION<em>STRING)[Settings.REVIEWS</em>DATABASE][
    Settings.REVIEWS_COLLECTION]</p>

<p>count = 0
done = 0
start = time.time()</p>

<p>with open(dataset_file) as dataset:
    count = sum(1 for line in dataset)</p>

<p>restaurant_dict = {}</p>

<p>with open(business<em>file) as businesses:
    next(businesses)
    for b in businesses:
        try:
            venue = json.loads(b)
        except ValueError:
            print 'Not Valid!!!!!'
        if 'Restaurants' in venue['categories']:
            restaurant</em>dict[venue['business_id']] = 1</p>

<p>with open(dataset<em>file) as dataset:
    next(dataset)
    for line in dataset:
        try:
            data = json.loads(line)
        except ValueError:
            print 'Oops!'
        if (data["type"] == "review") and (data["business</em>id"] in restaurant<em>dict):
            reviews</em>collection.insert({
                "reviewId": data["review<em>id"],
                "userId":  data["user</em>id"],
                "business": data["business_id"],
                "text": data["text"]
            })</p>

<pre><code>    done += 1
    if done % 100 == 0:
        end = time.time()
        os.system('cls')
        print 'Done ' + str(done) + ' out of ' + str(count) + ' in ' + str((end - start))
</code></pre>

<p>{% endhighlight %}</p>

<p>Part <strong>2</strong> is where we build the neural network model using word2vec
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
logger.setLevel(logging.INFO)</p>

<p>def get<em>businesses():
    businesses = {}
    reviews</em>collection = MongoClient(Settings.MONGO<em>CONNECTION</em>STRING)[Settings.REVIEWS<em>DATABASE][
            Settings.REVIEWS</em>COLLECTION]
    reviews<em>cursor = reviews</em>collection.find()
    for review in reviews_cursor:
        if review["business"] not in businesses: businesses[review["business"]] = 1</p>

<pre><code>return businesses.keys()
</code></pre>

<p>def load_stopwords():
    stopwords = {}
    with open('stopwords.txt', 'rU') as f:
        for line in f:
            stopwords[line.strip()] = 1</p>

<pre><code>return stopwords
</code></pre>

<p>def notstop(word):
    if word is not None:
        if word not in load_stopwords():
            if word not in string.punctuation:
                return word
    return word</p>

<p>def sent<em>list():
    stopwords = load</em>stopwords()
    businesses = {}
    sentence<em>list = []
    c = 0
    reviews</em>collection = MongoClient(Settings.MONGO<em>CONNECTION</em>STRING)[Settings.REVIEWS<em>DATABASE][Settings.REVIEWS</em>COLLECTION]
    reviews<em>cursor = reviews</em>collection.find()
    for review in reviews<em>cursor:
        businesses[review['business']] = 1
        sentences = nltk.sent</em>tokenize(review["text"].lower())
        for sentence in sentences:
            word<em>list = []
            for word in nltk.word</em>tokenize(sentence):
                if word not in load<em>stopwords():
                    if word not in string.punctuation:
                        if word.isalpha():
                            word</em>list.append(word)
            print len(businesses.keys())
            if len(word<em>list) != 0:
                sentence</em>list.append(word<em>list)
        if len(businesses.keys()) == 5000:
            break
        with open('businesses.txt', 'w') as f:
            for business in businesses.keys():
                f.write(business+'
')
    return sentence</em>list</p>

<p>def sent<em>Gen():
    stopwords = load</em>stopwords()
    businesses = {}</p>

<pre><code>c = 0
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
        if len(word_list) != 0 and len(businesses.keys()) &lt;= 1:
            yield word_list
</code></pre>

<p>print 'Done building the sentence generator'</p>

<p>sentences = sent_list()</p>

<h1>sentences = sent_Gen()</h1>

<p>skpg = 4</p>

<h1>phrasemodel = Phrases(sentences,min_count=10)</h1>

<p>model = Word2Vec(sentences,min_count=10,size=50,workers=4,sg=skpg)</p>

<h1>os.system("touch models/yelp<em>phrase</em>model")</h1>

<p>os.system("touch models/yelp<em>model"+str(skpg))
model.save('models/yelp</em>model'+str(skpg))</p>

<h1>phrasemodel.save('models/yelp<em>phrase</em>model')</h1>

<p>{% endhighlight %}</p>

<p>Part <strong>3</strong> has two parts. The first is the phrase extraction algorithm using a context free grammar to identify noun phrases probabilistically, and the second is where we find out what each user is talking about in each review (entity extraction), the program extracts entities for each review and puts it into a datbase asynchronously</p>

<p>{% highlight python %}</p>

<h1>coding=UTF-8</h1>

<p>import nltk
from nltk.corpus import brown</p>

<h1>This is a fast and simple noun phrase extractor (based on NLTK)</h1>

<h1>Feel free to use it, just keep a link back to this post</h1>

<h1>http://thetokenizer.com/2013/05/09/efficient-way-to-extract-the-main-topics-of-a-sentence/</h1>

<h1>Create by Shlomi Babluki</h1>

<h1>May, 2013</h1>

<h1>This is our fast Part of Speech tagger</h1>

<h6>#</h6>

<p>brown<em>train = brown.tagged</em>sents(categories='reviews')
regexp<em>tagger = nltk.RegexpTagger(
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
unigram</em>tagger = nltk.UnigramTagger(brown<em>train, backoff=regexp</em>tagger)
bigram<em>tagger = nltk.BigramTagger(brown</em>train, backoff=unigram_tagger)</p>

<h6>#</h6>

<h1>This is our semi-CFG; Extend it according to your own needs</h1>

<h6>#</h6>

<p>cfg = {}
cfg["NNP+NNP"] = "NNP"
cfg["NN+NN"] = "NNI"
cfg["NNI+NN"] = "NNI"
cfg["JJ+JJ"] = "JJ"
cfg["JJ+NN"] = "NNI"</p>

<h6>#</h6>

<p>class NPExtractor(object):</p>

<pre><code>def __init__(self, sentence):
    self.sentence = sentence

# Split the sentence into singlw words/tokens
def tokenize_sentence(self, sentence):
    tokens = nltk.word_tokenize(sentence)
    return tokens

# Normalize brown corpus' tags ("NN", "NN-PL", "NNS" &gt; "NN")
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
</code></pre>

<p>{% endhighlight %}</p>

<p>{% highlight python %}
import multiprocessing
import time
import sys
import phrase_extraction
import nltk
from pymongo import MongoClient</p>

<p>from settings import Settings</p>

<p>def load_stopwords():
    stopwords = {}
    with open('stopwords.txt', 'rU') as f:
        for line in f:
            stopwords[line.strip()] = 1</p>

<pre><code>return stopwords
</code></pre>

<p>def worker(identifier, skip, count):
    done = 0
    start = time.time()</p>

<pre><code>stopwords = load_stopwords()
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
</code></pre>

<p>def main():
    reviews<em>collection = MongoClient(Settings.MONGO</em>CONNECTION<em>STRING)[Settings.REVIEWS</em>DATABASE][
        Settings.REVIEWS<em>COLLECTION]
    reviews</em>cursor = reviews<em>collection.find()
    count = reviews</em>cursor.count()
    workers = 8
    batch = count / workers</p>

<pre><code>jobs = []
for i in range(workers):
    p = multiprocessing.Process(target=worker, args=((i + 1), i * batch, count / workers))
    jobs.append(p)
    p.start()

for j in jobs:
    j.join()
    print '%s.exitcode = %s' % (j.name, j.exitcode)
</code></pre>

<p>if <strong>name</strong> == '<strong>main</strong>':
    main()
{% endhighlight %}</p>

<p>Parts <strong>4</strong> and <strong>5</strong> are pretty much together as we gather the sentiment of each entity and based on the neural network we recommend venues according to the sentiment of a restaurant and how contextually similar their reviews are given by other users
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
import random</p>

<p>class ngram(dict):
    """Implementation of perl's autovivification feature."""
    def <strong>getitem</strong>(self, item):
        try:
            return dict.<strong>getitem</strong>(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value</p>

<h1>likelihood of recommendation</h1>

<p>def strippunc(s):
    out = s.translate(string.maketrans("",""), string.punctuation)
    return out</p>

<p>def load_stopwords():
    stopwords = {}
    with open('stopwords.txt', 'rU') as f:
        for line in f:
            stopwords[line.strip()] = 1</p>

<pre><code>return stopwords
</code></pre>

<p>def notstop(word):
    if word is not None:
        if word not in load_stopwords():
            if word not in string.punctuation:
                return word
    return word</p>

<p>def sublistExists(list, sublist):
    for i in range(len(list)-len(sublist)+1):
        if sublist == list[i:i+len(sublist)]:
            return True #return position (i) if you wish
    return False</p>

<p>def intersection(input<em>list):
    return reduce(set.intersection,map(set,input</em>list)) </p>

<p>def recommendation(userId,n):
    model = gensim.models.Word2Vec.load('models/yelp<em>model4')
    tags</em>collection = MongoClient(Settings.MONGO<em>CONNECTION</em>STRING)[Settings.TAGS<em>DATABASE][
        Settings.REVIEWS</em>COLLECTION]
    reviews<em>collection = MongoClient(Settings.MONGO</em>CONNECTION<em>STRING)[Settings.REVIEWS</em>DATABASE][Settings.REVIEWS<em>COLLECTION]
    users = ngram()
    ep = []
    most</em>sim = []
    restaurants = []</p>

<pre><code>rec_list= []
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

            if len(intersection([str(sentence).lower().split(),phrase.split()])) &gt; 1 and np.mean(scores) &gt; 0:
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
    if len(intersection([wordcloud,review["text"].lower().split()])) &gt;= 5  and np.mean(raw_scores) &gt; .2:
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
</code></pre>

<p>print recommendation("zvNimI98mrmhgNOOrzOiGg",10)</p>

<p>{% endhighlight %}</p>

<p><strong>Results comparison</strong></p>

<p>The following is an implementation of a collaborative filtering model and you can test the models to compare the results for research puprposes</p>

<p>{% highlight python %}
from scrapy.spider import BaseSpider
from scrapy.selector import HtmlXPathSelector
from scrapy.http import Request
import re</p>

<p>from pandoraFood.items import Review</p>

<h1>url string components for reviewer pages</h1>

<p>URL<em>BASE = 'http://www.yelp.com/user</em>details<em>reviews</em>self?userid='
FILTER<em>SETTINGS = '&amp;review</em>filter=category&amp;category_filter=restaurants'</p>

<h1>yelp unique url endings for each restaurant</h1>

<p>RESTAURANTS = ['z-and-y-restaurant-san-francisco', \
               'koi-palace-daly-city', \
               'ino-sushi-san-francisco', \
               'blackwood-san-francisco-3']</p>

<p>def createRestaurantPageLinks(self, response):
   reviewsPerPage = 40
   hxs = HtmlXPathSelector(response)
   totalReviews = int(hxs.select('//h2[@id="total_reviews"]/text()').extract()[0].strip().split(' ')[0])
   pages = [Request(url=response.url + '?start=' + str(reviewsPerPage*(n+1)), \
                    callback=self.parse) \
            for n in range(totalReviews/reviewsPerPage)]
   return pages</p>

<p>def createReviewerPageLinks(self, response):
   reviewsPerPage = 10
   hxs = HtmlXPathSelector(response)
   totalReviews = int(hxs.select('//div[@id="review<em>lister</em>header"]/em/text()').extract()[0].split(' ')[0])
   pages = [Request(url=response.url + '&amp;rec_pagestart=' + str(reviewsPerPage*(n+1)), \
                    callback=self.parseReviewer) \
            for n in range(totalReviews/reviewsPerPage)]
   return pages</p>

<p>class RestaurantSpider(BaseSpider):
   name = 'crawlRestaurants'
   allowed<em>domains = ['yelp.com']
   start</em>urls = [ 'http://www.yelp.com/biz/%s' % s for s in RESTAURANTS]</p>

<p># default parse used for the landing page for each start_url
   def parse(self, response):
      requests = []</p>

<pre><code>  # extract all reviews from the page and return a list of requests for the 5 star reviewers' profiles
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
</code></pre>

<p># parse a given reviewer
   def parseReviewer(self, response):
      hxs = HtmlXPathSelector(response)
      restaurantUrls = hxs.select('//div[@class="review clearfix"]/ \
                                  div[@class="biz_info"]/h4/a/@href').extract()
      restaurants = [re.search(r'(?&lt;=/biz/)[^#]<em>', rest).group() for rest in restaurantUrls]
      reviewerName = hxs.select('//title/text()').extract()[0].split('|')[0].replace('\'s Profile','').strip()
      reviewerUserID = re.search(r'(?&lt;=userid=)[^&amp;]</em>', response.url).group()
      ratingText = hxs.select('//div[@class="rating"]/i/@title').extract()
      ratings = [s.replace(' star rating','') for s in ratingText]</p>

<pre><code>  reviews = []
  for i in range(len(restaurants)):
     review = Review()
     review['restaurant'] = restaurants[i]
     review['reviewerName'] = reviewerName
     review['reviewerUserID'] = reviewerUserID
     review['rating'] = float(ratings[i])
     reviews.append(review)

  # request additional pages if we are on page 1 of the reviewer
  additionalPages = []
  if response.url.find('&amp;rec_pagestart=') == -1:
     additionalPages = createReviewerPageLinks(self, response)

  return reviews + additionalPages
</code></pre>

<p>{% endhighlight %}</p>

<p><strong>Concluding Thoughts</strong></p>

<p>I know what you are thinking! How do we know this pile of code is even good. The real answer is <strong>We Don't</strong>! This algorithm is almost totally unsupervised and truthfully impossible to test. One thing we do know is the results when read manually, do appear to be relevant and potential solutions to a recommendation problem based on the premise I have given and the style with which I chose to tackle this problem. Reviews for one thing are inherently biased and almost impossible to scientifically break down, so this will be a forever work in progress.</p>

<p><strong>Potential Improvements</strong></p>

<p>I want to come up with a review food corpus so that I can recognize food entities specifically and target only food entities to clean up the recommender.</p>

<p><strong>Editorial Notes/Me rambling about random stuff that you should probably maybe read</strong> </p>

<p>There are certain parameters in this program that can be changed to alter results. The results will vary everytime the algorithm is run but the selection criteria will remain the same. Permuting the selection will affect results, not necessarily a bad thing. It will be interesting to explore this. You can manipulate various thresholds such as the sentiment or similarity thresholds to tailor the algorithms criteria to your needs. For those wondering, the algorithm does yield coherent results, and the results will be further studied.</p>

---
