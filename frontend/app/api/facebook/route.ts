import { NextResponse } from 'next/server';

// In-memory cache to prevent burning through Apify free credits
let cachedPosts: any = null;
let lastFetchTime: number = 0;
const CACHE_DURATION = 24 * 60 * 60 * 1000; // 24 hours

// Fallback data to show instantly while Apify takes 2-5 minutes to boot up and scrape
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

  // If we have cached data less than 24 hours old, return it instantly!
  if (cachedPosts && (Date.now() - lastFetchTime < CACHE_DURATION)) {
    return NextResponse.json(cachedPosts);
  }

  // Kick off background scrape to Apify without awaiting it (so we don't timeout the UI)
  triggerBackgroundScrape(token);

  // Return the cache (even if slightly expired) or fallback data instantly
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
      cachedPosts = formattedPosts;
      lastFetchTime = Date.now();
      console.log("Successfully cached new Facebook posts from Apify!");
    }
  } catch (err) {
    console.error("Background scrape error:", err);
  }
}
