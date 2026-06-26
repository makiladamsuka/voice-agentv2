import urllib.request, json
req = urllib.request.Request('https://facebook-pages-scraper2.p.rapidapi.com/get_facebook_posts_details?link=https%3A%2F%2Fwww.facebook.com%2Ffitmoments&timezone=UTC', headers={
    'x-rapidapi-host': 'facebook-pages-scraper2.p.rapidapi.com',
    'x-rapidapi-key': 'd2fd4b624cmshe9a81b16e079328p1524e6jsnf99691f50788'
})
res = urllib.request.urlopen(req).read()
data = json.loads(res)
if isinstance(data, list) and len(data) > 0:
    print('is list')
    print(list(data[0].keys()))
    if 'text' in data[0]: print(data[0]['text'][:20])
    elif 'post_text' in data[0]: print(data[0]['post_text'][:20])
elif isinstance(data, dict):
    print('is dict')
    print(list(data.keys()))
    if 'data' in data: print(type(data['data']))

