import urllib.request, json
req = urllib.request.Request('https://facebook-pages-scraper2.p.rapidapi.com/get_facebook_posts_details?link=https%3A%2F%2Fwww.facebook.com%2Ffitmoments&timezone=UTC', headers={
    'x-rapidapi-host': 'facebook-pages-scraper2.p.rapidapi.com',
    'x-rapidapi-key': 'd2fd4b624cmshe9a81b16e079328p1524e6jsnf99691f50788'
})
res = urllib.request.urlopen(req).read()
data = json.loads(res)
def find_strings(obj, path=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            find_strings(v, f'{path}.{k}')
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if i == 0:
                find_strings(v, f'{path}[0]')
    elif isinstance(obj, str) and len(obj) > 20:
        print(f'{path}: {obj[:40]}...')

if data.get('data') and data['data'].get('posts'):
    find_strings(data['data']['posts'][0], 'post')

