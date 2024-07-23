// pages/_app.js
import Head from 'next/head';
import '../styles/globals.css';

function MyApp({ Component, pageProps }) {
  return (
    <>
      <Head>
        <title>Akhil | Personal Portfolio</title>
        <meta name="description" content="Akhil is a full-stack developer." />
      </Head>
      <Component {...pageProps} />
    </>
  );
}

export default MyApp;
