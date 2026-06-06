import { NextResponse } from 'next/server';

export async function GET() {
  const pageId = process.env.FACEBOOK_PAGE_ID;
  const accessToken = process.env.FACEBOOK_ACCESS_TOKEN;

  if (!pageId || !accessToken) {
    return NextResponse.json({ error: 'Missing Facebook credentials' }, { status: 500 });
  }

  try {
    const url = `https://graph.facebook.com/v19.0/${pageId}/posts?fields=full_picture,message,created_time&limit=15&access_token=${accessToken}`;
    const response = await fetch(url);
    const data = await response.json();

    if (data.error) {
      console.error("Facebook API Error:", data.error);
      return NextResponse.json({ error: data.error.message }, { status: 500 });
    }

    // Filter to only posts that have a picture
    const postsWithPictures = data.data
      .filter((post: any) => post.full_picture && post.message)
      .slice(0, 5); // Take the top 5

    return NextResponse.json(postsWithPictures);
  } catch (error) {
    console.error("Failed to fetch Facebook posts:", error);
    return NextResponse.json({ error: 'Failed to fetch posts' }, { status: 500 });
  }
}
