import { LanguageProvider } from "@/contexts/LanguageContext";
import type { Metadata } from "next";
import { ThemeProvider } from "next-themes";
import "./globals.css";

// 暂时移除 Google Fonts 以避免 Turbopack 兼容性问题
// 使用系统字体作为替代
const fontVariables = {
  "--font-geist-sans": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
  "--font-serif-jp": "Georgia, 'Times New Roman', serif",
  "--font-geist-mono": "'Courier New', Consolas, Monaco, monospace"
};

export const metadata: Metadata = {
  title: "Deepwiki Open Source | Sheing Ng",
  description: "Created by Sheing Ng",
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className="antialiased"
        style={fontVariables}
      >
        <ThemeProvider attribute="data-theme" defaultTheme="system" enableSystem>
          <LanguageProvider>
            {children}
          </LanguageProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
