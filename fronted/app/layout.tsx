import type {Metadata} from 'next';
import './globals.css'; // Global styles

export const metadata: Metadata = {
  title: 'SatQuery',
  description: 'AI-powered satellite intelligence workstation prototype.',
  openGraph: {
    title: 'SatQuery',
    description: 'AI-powered satellite intelligence workstation prototype.',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'SatQuery',
    description: 'AI-powered satellite intelligence workstation prototype.',
  },
};

export default function RootLayout({children}: {children: React.ReactNode}) {
  return (
    <html lang="en">
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
