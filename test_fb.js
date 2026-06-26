const fs = require('fs');
try {
  const data = JSON.parse(fs.readFileSync('.facebook-cache.json', 'utf8'));
  console.log(JSON.stringify(data.posts[0], null, 2));
} catch (e) { console.error(e.message); }
