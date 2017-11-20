def twitter_search(authentication, path, start_date, final_date, word):
#Authentication is a list cointaining al the parameters setup
    
    import tweepy
    import csv
    import re
    import time
    

    
    
    #autentication parameters
    access_token = authentication[0]
    access_token_secret = authentication[1]
    consumer_key = authentication[2]
    consumer_secret = authentication[3]
    
    # autentication
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    
    
    # Open/Create a file to append data
    csvFile = open(path, 'a')
    
    #Use csv Writer
    csvWriter = csv.writer(csvFile, delimiter='\t' )
   
    cont = 1;
    #Tweet searching
    current_date = final_date;
    while current_date !=start_date:
        try:
            for tweet in tweepy.Cursor(api.search,
                                       q=word,
                                       since_id=start_date,
                                       until=current_date,
                                       count="1000",
                                       lang = 'en').items():
                if  ('RT @' not in tweet.text): #elimination of the retweets
                   tweet.text = re.sub(r"http\S+","",tweet.text); #elimination of the URLs in tweets
                   #print('\nDate: ', tweet.created_at, '\nAuthor of the tweet: ', tweet.user.name, '\n', tweet.text, '\nNumber of retweets: ', tweet.retweet_count)
                   csvWriter.writerow([tweet.retweet_count, tweet.user.name.encode('utf-8'), tweet.created_at, tweet.text.encode('utf-8')])
                   cont = cont +1;
                if(cont == 1000000):
                    current_date = time_decrease(current_date)
                    cont = 0
                    break
        except tweepy.TweepError:
            pass
     
def time_decrease(date):
    from datetime import datetime
    from datetime import timedelta
    date = datetime.strptime( date, '%Y-%m-%d');
    delta = timedelta(days=1);
    new_date = date-delta;
    final_date = datetime.isoformat(new_date)[0:10];
    return final_date
    
    
    
