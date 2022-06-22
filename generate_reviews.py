import pandas as pd
from google_play_scraper import Sort, reviews

apps_ids = ['com.bk.pt', 'com.ubercab.eats', 'com.mcdonalds.mobileapp', 'com.app.tgtg', 'com.lafourchette.lafourchette', 'com.bolt.deliveryclient', 'com.outdarelab.zomato', 'com.groupeseb.moulinex.food', 'pt.yunit.mobile.android.h3', 'com.telepizza']

ids = [apps_ids[1]]

apps_reviews = []

for id in ids:
    for score in list(range(1, 6)):
        rvs, _ = reviews(id,lang='pt', country='pt',
        sort=Sort. MOST_RELEVANT, count=5,
        filter_score_with=score)
        for r in rvs:
            r['appId'] = id
        apps_reviews.extend(rvs)

df = pd.DataFrame(apps_reviews)
print(df.head())
df.to_csv('reviews.csv', index=None, header=True)
