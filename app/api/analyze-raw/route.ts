import { NextResponse } from "next/server";
import { processAndSaveArticle } from "@/lib/inference/inference-service";

export async function POST(req: Request) {
  try {
    const { articleData } = await req.json();
    const result = await processAndSaveArticle(articleData);
    return NextResponse.json({ success: true, result });
  } catch (error) {
    return NextResponse.json({ error: "Fail" }, { status: 500 });
  }
}
