import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';

const CACHE_FILE = path.join(process.cwd(), '.facebook-cache.json');
const CACHE_DURATION = 10 * 60 * 1000; // 10 minutes

// Fallback data is now ONLY shown the very first time the app is ever booted before the first scrape finishes
const fallbackData = [
  {
    id: "fitmoments_1",
    full_picture: "https://images.unsplash.com/photo-1571019614242-c5c5dee9f50b?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80",
    message: "Start your morning right! Join our sunrise yoga sessions every Tuesday at the main campus quad. Don't forget your mat! 🧘‍♀️✨ #FitMoments #CampusWellness",
    created_time: new Date().toISOString()
  }
];

export async function GET() {
  let cachedPosts = null;
  let lastFetchTime = 0;

  // Attempt to read the persisted cache from disk
  try {
    const fileContent = await fs.readFile(CACHE_FILE, 'utf-8');
    const parsed = JSON.parse(fileContent);
    cachedPosts = parsed.posts;
    lastFetchTime = parsed.timestamp;
  } catch (err) {
    // Cache file doesn't exist yet
  }

  const needsRefresh = !cachedPosts || (Date.now() - lastFetchTime > CACHE_DURATION);

  if (needsRefresh) {
    // Fetch fresh data synchronously so the response is immediately up to date
    await triggerRSSScrape();
    try {
      const fileContent = await fs.readFile(CACHE_FILE, 'utf-8');
      const parsed = JSON.parse(fileContent);
      cachedPosts = parsed.posts;
    } catch (err) {
      // Ignore read errors
    }
  }

  // Return the persisted real data instantly (or the fallback if this is the first boot ever)
  return NextResponse.json(cachedPosts || fallbackData);
}

async function triggerRSSScrape() {
  try {
    console.log("Starting background RSS scrape for fitmoments...");
    const response = await fetch("https://rss.app/feeds/PpO8cOM0sBcILogo.xml");
    
    if (!response.ok) {
        console.error("RSS API failed:", await response.text());
        return;
    }

    const xml = await response.text();
    
    // Parse the XML feed manually using Regex (since this is a simple structured format)
    const itemRegex = /<item>[\s\S]*?<\/item>/g;
    const items = xml.match(itemRegex) || [];
    
    const formattedPosts = items.slice(0, 5).map((item: string, index: number) => {
      // Extract Image
      const imgMatch = item.match(/<media:content[^>]+url="([^"]+)"/);
      let imgUrl = imgMatch ? imgMatch[1].replace(/&amp;/g, '&') : null;
      if (!imgUrl) {
         const descImgMatch = item.match(/<img src="([^"]+)"/);
         imgUrl = descImgMatch ? descImgMatch[1].replace(/&amp;/g, '&') : null;
      }
      
      // Extract Text
      const titleMatch = item.match(/<title><!\[CDATA\[([\s\S]*?)\]\]><\/title>/);
      const message = titleMatch ? titleMatch[1].trim() : "New update from FIT Moments!";
      
      // Extract Date
      const dateMatch = item.match(/<pubDate>([^<]+)<\/pubDate>/);
      const created_time = dateMatch ? new Date(dateMatch[1]).toISOString() : new Date().toISOString();
      
      return {
        id: `rss_${index}_${Date.now()}`,
        full_picture: imgUrl || "https://images.unsplash.com/photo-1571019614242-c5c5dee9f50b?ixlib=rb-4.0.3",
        message: message.length > 150 ? message.substring(0, 150) + "..." : message,
        created_time
      };
    });

    if (formattedPosts.length > 0) {
      // Persist the real posts to disk so they survive server restarts!
      await fs.writeFile(CACHE_FILE, JSON.stringify({
        timestamp: Date.now(),
        posts: formattedPosts
      }, null, 2));
      console.log("Successfully cached new Facebook posts from RSS.app to disk!");
    }
  } catch (err) {
    console.error("Background scrape error:", err);
  }
}
