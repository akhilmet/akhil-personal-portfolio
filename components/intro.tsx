"use client";

import Image from "next/image";
import React from "react";
import { motion } from "framer-motion";
import Link from "next/link";
import { BsArrowRight, BsLinkedin } from "react-icons/bs";
import { FaGithubSquare } from "react-icons/fa";
import { useSectionInView } from "@/lib/hooks";
import { useActiveSectionContext } from "@/context/active-section-context";
import { TypeAnimation } from 'react-type-animation';

export default function Intro() {
  const { ref } = useSectionInView("Home", 0.5);
  const { setActiveSection, setTimeOfLastClick } = useActiveSectionContext();

  return (
    <section
      ref={ref}
      id="home"
      className="relative h-screen flex items-center justify-center bg-gradient-to-br from-green-800 to-blue-900 text-white"
    >
      <div className="text-center">
        <motion.h1
          className="text-5xl font-bold mb-4"
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          Nice to meet you! 👋
        </motion.h1>
        <motion.h2
          className="text-6xl font-extrabold mb-4"
          initial={{ opacity: 0, y: -30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          I'm Akhil
        </motion.h2>
        <motion.div
          className="text-3xl font-semibold mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          A{" "}
          <TypeAnimation
            sequence={[
              'Software Engineer',
              1500,
              'Full-Stack Developer',
              1500,
              'Problem-Solver',
              1500,
            ]}
            wrapper="span"
            speed={50}
            className="font-bold text-yellow-300"
            repeat={Infinity}
          />
        </motion.div>
      </div>

      <motion.div
        className="absolute right-20 top-1/2 transform -translate-y-1/2"
        initial={{ opacity: 0, x: 50 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5, delay: 0.6 }}
      >
        <div className="relative group">
          <Image
            src="/profile-pic.png"
            alt="Akhil Metukuru"
            width={300}
            height={300}
            className="rounded-full object-cover transition-transform duration-300 group-hover:scale-105"
          />
          <div className="absolute inset-0 rounded-full bg-gradient-to-r from-blue-400 to-green-400 opacity-0 group-hover:opacity-30 transition-opacity duration-300" />
        </div>
      </motion.div>

      <nav className="absolute top-4 left-1/2 transform -translate-x-1/2">
        <ul className="flex space-x-4">
          {["Home", "About", "Skills", "Experience", "Projects", "Contact"].map((item) => (
            <li key={item}>
              <Link href={`#${item.toLowerCase()}`} className="text-white hover:text-yellow-300 transition-colors">
                {item}
              </Link>
            </li>
          ))}
        </ul>
      </nav>
    </section>
  );
}