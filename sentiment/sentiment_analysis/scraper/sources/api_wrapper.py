#!/usr/bin/env python
# encoding: utf-8

import tweepy  # https://github.com/tweepy/tweepy
import csv
import logging
import time

# This script was downloaded form : https://gist.github.com/yanofsky/5436496

# Twitter API credentials
'''
api_key = "xyWmHDviqN0R933YO02c99KmD"
api_key_secret = "TGhQp5kFkXzUueFLsL1DuNLTiVxfZQIFVD1Pqobm5ugQuwaxdu"
access_token = "1063516087751503872-e5aNVELMhHx5Y6i5Xy0RJVCkruupu9"
access_token_secret = "SM0FL1X3ki6m0fj3ehr6GxPnx4YtD8Mgk7VsSRKDF83qk"
'''


class Auth:
	def __init__(self, ck, cs, ak, acs):
		self.ck = ck
		self.cs = cs
		self.ak = ak
		self.acs = acs


class API:
	def __init__(self, key):
		self.key = key
		self.api = None
		self.authorize()

	def authorize(self):
		"""
		Function to set up the api
		:return:
		"""
		auth = tweepy.OAuthHandler(self.key.ck, self.key.cs)
		auth.set_access_token(self.key.ak, self.key.acs)
		self.api = tweepy.API(auth,wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)

	def get_all_tweets(self, uid, repeat = True):
		"""
		Collect all tweets of an user.
		:param uid: user id
		:param repeat: boolean to not get repeat tweets
		:return:
		"""

		# initialize a list to hold all the tweepy Tweets
		alltweets = []

		# make initial request for most recent tweets (200 is the maximum allowed count)
		new_tweets = []
		try:
			new_tweets = self.api.user_timeline(id=uid, count=200)
			alltweets.extend(new_tweets)
		except tweepy.error.TweepError as e:
			print('Exception while collecting tweet: {}'.format(e))
			self.authorize()

		if not repeat:
			return new_tweets
		# save the id of the oldest tweet less one
		oldest = alltweets[-1].id - 1 if alltweets else 0

		# keep grabbing tweets until there are no tweets left to grab
		while len(new_tweets) > 0:
			# print "getting tweets before %s, number of tweets retrieved in this request %s." % (oldest, len(new_tweets))
			# print 'first tweet id %s, last tweet id %s' %(new_tweets[0].id, new_tweets[len(new_tweets)-1].id)
			# all subsequent requests use the max_id param to prevent duplicates
			try:
				new_tweets = self.api.user_timeline(id=uid, count=200, max_id=oldest)
			except tweepy.error.RateLimitError as e:
				raise
			except tweepy.error.TweepError as e:
				print('Exception while collecting tweet: {}'.format(e))
				self.authorize()

			# save most recent tweets
			alltweets.extend(new_tweets)

			# update the id of the oldest tweet less one
			oldest = alltweets[-1].id - 1

			# print "...%s tweets downloaded so far" % (len(alltweets))

		return alltweets

	def get_user_profiles_from_ids(self, uids):
		"""
		Collect user profiles given user ids
		:param uids: userids
		:return: user objects
		"""
		return self.api.lookup_users(user_ids=uids)

	def get_user_profiles_from_names(self, unames):
		"""
		Collect user profiles given user names
		:param unames: profile usernames
		:return: uer objects
		"""
		return self.api.lookup_users(screen_names=unames)

	def get_friend_ids(self, uname):
		"""
		Collect ids of all the friends of an user
		:param uname: profile username
		:return: list of friend objects
		"""
		ids = []
		try:
			for page in tweepy.Cursor(self.api.friends_ids, screen_name=uname).pages():
				ids.extend(page)
				time.sleep(20)
		except Exception as e:
			print('Exception while collecting friends id:', e)
		return ids

	def get_follower_ids(self, uname):
		"""
		Collect ids of all the followers of an user
		:param uname: profile username
		:return: list of follower ids
		"""
		ids = []
		try:
			for page in tweepy.Cursor(self.api.followers_ids, screen_name=uname).pages():
				ids.extend(page)
				time.sleep(20)
		except Exception as e: print('Exception while collecting followers id:', e)
		return ids

	def get_tweets_from_ids(self, tweet_ids):
		"""
		Get all tweets given the ids
		:param tweet_ids: list of tweet ids
		:return: list of tweets from given ids
		"""
		return self.api.statuses_lookup(tweet_ids, include_entities=True, trim_user=False, tweet_mode="extended")

	def lists_subscribed(self, uname):
		"""
		function to get subscribed users of a given user
		:param uname: profile username
		:return: list of ids of subscribed users
		"""
		ids = []
		for page in tweepy.Cursor(self.api.lists_subscriptions, screen_name=uname).pages(): ids.extend(page)
		# time.sleep(20)
		return ids

	def lists_created(self, uname):
		return self.api.lists_all(uname)

	def lists_member(self, uname):
			ids = []
			for page in tweepy.Cursor(self.api.lists_memberships, screen_name=uname).pages(): ids.extend(page)
			# time.sleep(20)
			return ids

	def search_tweets(self, query, count, since):
		"""
		Search tweets based on query
		:param query: hashtag to search for
		:param count: number of pages to collect
		:param since: date to collect forward from
		:return: list of tweets based on query
		"""
		tweets = []
		for page in tweepy.Cursor(self.api.search, q=query, lang="en", result_type='popular', since=since).pages(count):
			tweets.extend(page)
		return tweets
