"use client";

import Image from "next/image";
import React from "react";
import { motion } from "framer-motion";
import Link from "next/link";
import { BsLinkedin } from "react-icons/bs";
import { FaInstagram } from "react-icons/fa";
import { TypeAnimation } from 'react-type-animation';

export default function Intro() {
  return (
    <section className="min-h-screen flex flex-col bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900">
      <nav className="p-4">
        <ul className="flex justify-center space-x-4">
          {["Home", "About", "Projects", "Skills", "Experience", "Contact"].map((item) => (
            <li key={item}>
              <Link href={`#${item.toLowerCase()}`} className="text-white hover:text-yellow-300 transition-colors">
                {item}
              </Link>
            </li>
          ))}
        </ul>
      </nav>

      <div className="flex-grow flex items-center justify-center">
        <div className="container mx-auto flex flex-col md:flex-row items-center justify-between px-4">
          <motion.div 
            className="text-white text-center md:text-left mb-8 md:mb-0"
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
          >
            <h1 className="text-4xl md:text-5xl font-bold mb-4">Hello, I'm Akhil Metukuru.</h1>
            <p className="text-xl md:text-2xl mb-4">
              I'm a rising junior at the University of Maryland studying Computer Science with a focus on{' '}
              <TypeAnimation
                sequence={[
                  'Artificial Intelligence',
                  2000,
                  'Machine Learning',
                  2000,
                  'Software Engineering',
                  2000,
                ]}
                wrapper="span"
                speed={50}
                className="font-bold text-yellow-300"
                repeat={Infinity}
              />
            </p>
            <div className="flex justify-center md:justify-start space-x-4">
              <Link href="#contact" className="bg-white text-blue-900 px-6 py-2 rounded-full hover:bg-yellow-300 transition-colors">
                Contact me here
              </Link>
              <a href="https://linkedin.com/in/yourprofile" target="_blank" rel="noopener noreferrer" className="text-white hover:text-yellow-300">
                <BsLinkedin size={24} />
              </a>
              <a href="https://instagram.com/yourprofile" target="_blank" rel="noopener noreferrer" className="text-white hover:text-yellow-300">
                <FaInstagram size={24} />
              </a>
            </div>
          </motion.div>

          <motion.div
            className="relative"
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Image
              src="/your-profile-pic.jpg"
              alt="Akhil Metukuru"
              width={300}
              height={300}
              className="rounded-full"
            />
            <span className="absolute bottom-0 right-0 text-4xl">👋</span>
          </motion.div>
        </div>
      </div>
    </section>
  );
}