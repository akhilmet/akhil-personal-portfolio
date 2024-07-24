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
      className="mb-28 max-w-[50rem] text-center sm:mb-0 scroll-mt-[100rem]"
    >
      <div className="relative">
        <motion.div
          initial={{ opacity: 0, scale: 0 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ type: "tween", duration: 0.2 }}
        >
          <Tilt className="inline-block" options={{ max: 25, scale: 1.05 }}>
            <Image
              src="/profile-pic.png"
              alt="Akhil Metukuru"
              width="192"
              height="192"
              quality="95"
              priority={true}
              className="rounded-full object-cover border-[0.35rem] border-white shadow-xl"
            />
          </Tilt>
        </motion.div>

        <motion.span
          className="absolute top-0 left-0 text-4xl"
          initial={{ opacity: 0, scale: 0 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{
            type: "spring",
            stiffness: 125,
            delay: 0.1,
            duration: 0.7,
          }}
        >
          👋
        </motion.span>
      </div>

      <motion.h1
        className="mb-10 mt-4 px-4 text-2xl font-medium !leading-[1.5] sm:text-4xl"
        initial={{ opacity: 0, y: 100 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <span 
          className="font-bold transition-all duration-300 ease-in-out"
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
          Nice to meet you! I'm Akhil Metukuru.
        </span>
        <br />
        I'm a{" "}
        <TypeAnimation
          sequence={[
            'Software Engineer.',
            2000,
            'Full-Stack Developer.',
            2000,
            'Problem-Solver.',
            2000,
          ]}
          wrapper="span"
          speed={50}
          className="font-bold underline"
          repeat={Infinity}
        />
      </motion.h1>

      {/* Rest of the component remains the same */}
    </section>
  );
}