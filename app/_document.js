// pages/_document.js
import { Html, Head, Main, NextScript } from 'next/document';

export default function Document() {
  return (
    <Html>
      <Head>
        <title>Akhil | Personal Portfolio</title>
        <meta name="description" content="Akhil is a full-stack developer." />
      </Head>
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
