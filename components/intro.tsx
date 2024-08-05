"use client";

import React, { useState } from 'react';
import Image from "next/image";
import { motion } from "framer-motion";
import Link from "next/link";
import { BsArrowRight, BsLinkedin } from "react-icons/bs";
import { FaInstagram } from "react-icons/fa";
import { useSectionInView } from "@/lib/hooks";
import { useActiveSectionContext } from "@/context/active-section-context";
import { TypeAnimation } from 'react-type-animation';
import { Tilt } from 'react-tilt';

export default function Intro() {
  const { ref } = useSectionInView("Home", 0.5);
  const { setActiveSection, setTimeOfLastClick } = useActiveSectionContext();
  const [isHovered, setIsHovered] = useState(false);

  return (
    <section
      ref={ref}
      id="home"
      className="h-screen flex items-center justify-center bg-gradient-to-br from-purple-900 to-pink-800"
    >
      <div className="max-w-[50rem] text-center text-white">
        <div className="flex justify-center mb-8">
          <Tilt className="Tilt" options={{ max: 25, scale: 1.05 }}>
            <Image
              src="/profile-pic.png"
              alt="Akhil Metukuru"
              width={200}
              height={200}
              quality="95"
              priority={true}
              className="rounded-full object-cover border-4 border-white shadow-xl"
            />
          </Tilt>
        </div>

        <motion.h2
          className="text-2xl font-medium mb-2"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          Nice to meet you! <span className="inline-block">👋</span>
        </motion.h2>
        
        <motion.h1
          className="text-5xl font-bold mb-4 relative"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
        >
          I'm Akhil Metukuru
          {isHovered && (
            <motion.span
              className="absolute inset-0 bg-gradient-to-r from-purple-400 to-pink-400"
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 0.3 }}
              exit={{ scale: 0, opacity: 0 }}
              transition={{ duration: 0.3 }}
            />
          )}
        </motion.h1>
        
        <motion.h3
          className="text-2xl mb-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          A{" "}
          <TypeAnimation
            sequence={[
              'Software Engineer.',
              1500,
              'Problem-Solver.',
              1500,
              'Tech Enthusiast.',
              1500,
            ]}
            wrapper="span"
            speed={60}
            className="font-bold"
            repeat={Infinity}
          />
        </motion.h3>

        <motion.div
          className="flex items-center justify-center gap-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <Link
            href="#contact"
            className="bg-white text-purple-900 px-7 py-3 flex items-center gap-2 rounded-full font-medium hover:bg-opacity-90 transition"
            onClick={() => {
              setActiveSection("Contact");
              setTimeOfLastClick(Date.now());
            }}
          >
            Contact me here{" "}
            <BsArrowRight className="opacity-70 group-hover:translate-x-1 transition" />
          </Link>

          <a
            href="https://linkedin.com/in/akmet"
            target="_blank"
            rel="noopener noreferrer"
            className="bg-white/10 p-4 text-white rounded-full hover:bg-white/20 transition"
          >
            <BsLinkedin />
          </a>

          <a
            href="https://instagram.com/yourusername"
            target="_blank"
            rel="noopener noreferrer"
            className="bg-white/10 p-4 text-white rounded-full hover:bg-white/20 transition"
          >
            <FaInstagram />
          </a>
        </motion.div>
      </div>
    </section>
  );
}