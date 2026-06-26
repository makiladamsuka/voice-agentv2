import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET(req: NextRequest) {
  try {
    const url = new URL(req.url);
    const imagePath = url.searchParams.get("path");

    if (!imagePath) {
      return new NextResponse("Missing path parameter", { status: 400 });
    }

    // Path traversal protection
    if (imagePath.includes("..") || imagePath.startsWith("/")) {
      return new NextResponse("Invalid path", { status: 403 });
    }

    const backendDir = process.env.BACKEND_DIR
      ? path.resolve(process.env.BACKEND_DIR)
      : path.join(process.cwd(), "..", "backend");
    const fullPath = path.join(backendDir, "assets", imagePath);

    if (!fs.existsSync(fullPath)) {
      return new NextResponse("File not found", { status: 404 });
    }

    const fileBuffer = fs.readFileSync(fullPath);

    let contentType = "application/octet-stream";
    const ext = path.extname(fullPath).toLowerCase();
    if (ext === ".jpg" || ext === ".jpeg") contentType = "image/jpeg";
    else if (ext === ".png") contentType = "image/png";
    else if (ext === ".webp") contentType = "image/webp";
    else if (ext === ".gif") contentType = "image/gif";

    return new NextResponse(fileBuffer, {
      status: 200,
      headers: {
        "Content-Type": contentType,
        "Cache-Control": "public, max-age=31536000, immutable",
      },
    });
  } catch (error: any) {
    console.error("Image API error:", error);
    return new NextResponse("Internal Server Error", { status: 500 });
  }
}
