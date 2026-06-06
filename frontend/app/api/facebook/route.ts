import { NextResponse } from 'next/server';

export async function GET() {
  // Since this is Scenario B (you do not own the 'fitmoments' page), 
  // the official Graph API blocks us from reading it without App Review.
  // For demonstration purposes on your kiosk, we are returning realistic mock data
  // that perfectly matches the format of the Graph API!
  
  const mockFitMomentsPosts = [
    {
      id: "fitmoments_1",
      full_picture: "https://images.unsplash.com/photo-1571019614242-c5c5dee9f50b?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80",
      message: "Start your morning right! Join our sunrise yoga sessions every Tuesday at the main campus quad. Don't forget your mat! 🧘‍♀️✨ #FitMoments #CampusWellness",
      created_time: new Date().toISOString()
    },
    {
      id: "fitmoments_2",
      full_picture: "https://images.unsplash.com/photo-1534438327276-14e5300c3a48?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80",
      message: "New equipment alert! 🚨 We just upgraded the cardio section in the student rec center. Come try out the new smart treadmills today.",
      created_time: new Date(Date.now() - 86400000).toISOString() // 1 day ago
    },
    {
      id: "fitmoments_3",
      full_picture: "https://images.unsplash.com/photo-1526506159807-1a52de1d52a7?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80",
      message: "Congratulations to the winners of our Spring Campus Marathon! 🏃‍♂️🏆 You all crushed it out there. Check the link in our bio for full race times.",
      created_time: new Date(Date.now() - 172800000).toISOString() // 2 days ago
    }
  ];

  return NextResponse.json(mockFitMomentsPosts);
}
