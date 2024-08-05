"use client";

import React, { useState } from 'react';
import Image from "next/image";
import { motion } from "framer-motion";
import Link from "next/link";
import { BsArrowRight, BsLinkedin } from "react-icons/bs";
import { FaGithubSquare } from "react-icons/fa";
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
      className="mb-28 max-w-[50rem] sm:mb-0 scroll-mt-[100rem] px-4"
    >
      <div className="flex flex-col-reverse md:flex-row items-center justify-between">
        <div className="text-left md:w-2/3">
          <motion.h2
            className="text-2xl font-medium mb-2"
            initial={{ opacity: 0, y: 100 }}
            animate={{ opacity: 1, y: 0 }}
          >
            Nice to meet you! <span className="inline-block">👋</span>
          </motion.h2>
          
          <motion.h1
            className="text-4xl md:text-5xl font-bold mb-4"
            initial={{ opacity: 0, y: 100 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <span 
              className="transition-all duration-300 ease-in-out"
              style={{
                background: isHovered ? 'linear-gradient(to right, #4ade80, #3b82f6)' : 'none',
                WebkitBackgroundClip: isHovered ? 'text' : 'none',
                WebkitTextFillColor: isHovered ? 'transparent' : 'inherit',
                padding: isHovered ? '0 8px' : '0',
                borderRadius: isHovered ? '4px' : '0',
              }}
              onMouseEnter={() => setIsHovered(true)}
              onMouseLeave={() => setIsHovered(false)}
            >
              I'm Akhil Metukuru
            </span>
          </motion.h1>
          
          <motion.h3
            className="text-xl md:text-2xl mb-6"
            initial={{ opacity: 0, y: 100 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            I'm a{" "}
            <TypeAnimation
              sequence={[
                'Software Engineer.',
                1400,
                'Full-Stack Developer.',
                1400,
                'Problem-Solver.',
                1400,
              ]}
              wrapper="span"
              speed={50}
              className="font-bold"
              repeat={Infinity}
            />
          </motion.h3>

          <motion.div
            className="flex items-center gap-2 text-lg font-medium"
            initial={{ opacity: 0, y: 100 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <Link
              href="#contact"
              className="group bg-gray-900 text-white px-7 py-3 flex items-center gap-2 rounded-full outline-none focus:scale-110 hover:scale-110 hover:bg-gray-950 active:scale-105 transition"
              onClick={() => {
                setActiveSection("Contact");
                setTimeOfLastClick(Date.now());
              }}
            >
              Contact me here{" "}
              <BsArrowRight className="opacity-70 group-hover:translate-x-1 transition" />
            </Link>

            <a
              className="bg-white p-4 text-gray-700 hover:text-gray-950 flex items-center gap-2 rounded-full focus:scale-[1.15] hover:scale-[1.15] active:scale-105 transition cursor-pointer borderBlack dark:bg-white/10 dark:text-white/60"
              href="https://linkedin.com/in/akmet"
              target="_blank"
              rel="noopener noreferrer"
            >
              <BsLinkedin />
            </a>

            <a
              className="bg-white p-4 text-gray-700 flex items-center gap-2 text-[1.35rem] rounded-full focus:scale-[1.15] hover:scale-[1.15] hover:text-gray-950 active:scale-105 transition cursor-pointer borderBlack dark:bg-white/10 dark:text-white/60"
              href="https://github.com/akhilmet"
              target="_blank"
              rel="noopener noreferrer"
            >
              <FaGithubSquare />
            </a>
          </motion.div>
        </div>

        <motion.div
          className="md:w-1/3 mb-8 md:mb-0"
          initial={{ opacity: 0, scale: 0 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ type: "tween", duration: 0.2 }}
        >
          <Tilt className="inline-block" options={{ max: 25, scale: 1.05 }}>
            <Image
              src="/profile-pic.png"
              alt="Akhil Metukuru"
              width={300}
              height={300}
              quality="95"
              priority={true}
              className="rounded-full object-cover border-[0.35rem] border-white shadow-xl"
            />
          </Tilt>
        </motion.div>
      </div>
    </section>
  );
}