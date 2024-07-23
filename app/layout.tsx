import Header from "@/components/header";
import "./globals.css";
import { Inter } from "next/font/google";
import ActiveSectionContextProvider from "@/context/active-section-context";
import Footer from "@/components/footer";
import ThemeSwitch from "@/components/theme-switch";
import ThemeContextProvider from "@/context/theme-context";
import { Toaster } from "react-hot-toast";
import Head from "next/head";
import { BsLinkedin, BsGithub, BsEnvelope } from 'react-icons/bs';

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "Akhil | Personal Portfolio",
  description: "Akhil is a full-stack developer.",
};

const SideLinks = () => (
  <>
    <div className="fixed left-4 bottom-0 flex flex-col items-center z-50">
      <a href="https://linkedin.com/in/akmet" target="_blank" rel="noopener noreferrer" className="mb-4 text-gray-700 hover:text-gray-900 dark:text-gray-300 dark:hover:text-white">
        <BsLinkedin size={24} />
      </a>
      <a href="https://github.com/akhilmet" target="_blank" rel="noopener noreferrer" className="mb-4 text-gray-700 hover:text-gray-900 dark:text-gray-300 dark:hover:text-white">
        <BsGithub size={24} />
      </a>
      <div className="w-0.5 h-24 bg-gray-300 dark:bg-gray-700"></div>
    </div>
    <div className="fixed right-4 bottom-0 flex flex-col items-center z-50">
      <a href="mailto:akhil.metukuru2016@gmail.com" className="vertical-text mb-4 text-gray-700 hover:text-gray-900 dark:text-gray-300 dark:hover:text-white">
        <BsEnvelope size={24} />
      </a>
      <div className="w-0.5 h-24 bg-gray-300 dark:bg-gray-700"></div>
    </div>
  </>
);

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="!scroll-smooth">
      <Head>
        <title>{metadata.title}</title>
        <meta name="description" content={metadata.description} />
      </Head>
      <body
        className={`${inter.className} bg-gray-50 text-gray-950 relative pt-28 sm:pt-36 dark:bg-gray-900 dark:text-gray-50 dark:text-opacity-90`}
      >
        <div className="bg-[#fbe2e3] absolute top-[-6rem] -z-10 right-[11rem] h-[31.25rem] w-[31.25rem] rounded-full blur-[10rem] sm:w-[68.75rem] dark:bg-[#946263]"></div>
        <div className="bg-[#dbd7fb] absolute top-[-1rem] -z-10 left-[-35rem] h-[31.25rem] w-[50rem] rounded-full blur-[10rem] sm:w-[68.75rem] md:left-[-33rem] lg:left-[-28rem] xl:left-[-15rem] 2xl:left-[-5rem] dark:bg-[#676394]"></div>

        <ThemeContextProvider>
          <ActiveSectionContextProvider>
            <Header />
            <SideLinks />
            <main className="px-4 sm:px-8 md:px-16 lg:px-24">
              {children}
            </main>
            <Footer />

            <Toaster position="top-right" />
            <ThemeSwitch />
          </ActiveSectionContextProvider>
        </ThemeContextProvider>
      </body>
    </html>
  );
}