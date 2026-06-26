import urllib.request, json
req = urllib.request.Request('https://facebook-pages-scraper2.p.rapidapi.com/get_facebook_posts_details?link=https%3A%2F%2Fwww.facebook.com%2Ffitmoments&timezone=UTC', headers={
    'x-rapidapi-host': 'facebook-pages-scraper2.p.rapidapi.com',
    'x-rapidapi-key': 'd2fd4b624cmshe9a81b16e079328p1524e6jsnf99691f50788'
})
res = urllib.request.urlopen(req).read()
data = json.loads(res)
if isinstance(data, list) and len(data) > 0:
    print('basic_info keys:', list(data[0]['basic_info'].keys()))
    if 'title' in data[0]['basic_info']: print('title:', data[0]['basic_info']['title'][:50])
    if 'text' in data[0]['basic_info']: print('text:', data[0]['basic_info']['text'][:50])
    print('attachments:', list(data[0].get('attachments', {}).keys()))
    all_subs = data[0].get('attachments', {}).get('all_subattachments', {}).get('nodes', [])
    if len(all_subs) > 0:
        print('img uri:', all_subs[0].get('media', {}).get('image', {}).get('uri'))

