import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AIT-204 | RNN Temperature Forecasting",
  description:
    "Interactive RNN temperature forecasting demo — AIT-204 Deep Learning assignment",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
