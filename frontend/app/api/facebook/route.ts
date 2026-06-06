import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';

const CACHE_FILE = path.join(process.cwd(), '.facebook-cache.json');
const CACHE_DURATION = 24 * 60 * 60 * 1000; // 24 hours

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
  const token = process.env.APIFY_API_TOKEN;

  if (!token) {
    return NextResponse.json(fallbackData);
  }

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
    // Kick off background scrape to Apify
    triggerBackgroundScrape(token);
  }

  // Return the persisted real data instantly (or the fallback if this is the first boot ever)
  return NextResponse.json(cachedPosts || fallbackData);
}

async function triggerBackgroundScrape(token: string) {
  try {
    console.log("Starting background Apify scrape for fitmoments...");
    const response = await fetch(`https://api.apify.com/v2/acts/apify~facebook-pages-scraper/run-sync-get-dataset-items?token=${token}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        startUrls: [{ url: "https://www.facebook.com/fitmoments" }],
        resultsLimit: 5
      })
    });
    
    if (!response.ok) {
        console.error("Apify API failed:", await response.text());
        return;
    }

    const data = await response.json();
    
    // Map Apify output to match our UI expectations
    const formattedPosts = data
      .filter((post: any) => post.photos && post.photos.length > 0)
      .map((post: any, index: number) => ({
        id: `apify_${index}`,
        full_picture: post.photos[0],
        message: post.text || "",
        created_time: post.time || new Date().toISOString()
      }));

    if (formattedPosts.length > 0) {
      // Persist the real posts to disk so they survive server restarts!
      await fs.writeFile(CACHE_FILE, JSON.stringify({
        timestamp: Date.now(),
        posts: formattedPosts
      }, null, 2));
      console.log("Successfully cached new Facebook posts from Apify to disk!");
    }
  } catch (err) {
    console.error("Background scrape error:", err);
  }
}
