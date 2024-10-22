"use client";

import React from "react";
import SectionHeading from "./section-heading";
import { motion } from "framer-motion";
import { useSectionInView } from "@/lib/hooks";

export default function About() {
  const { ref } = useSectionInView("About");

  return (
    <motion.section
      ref={ref}
      className="mb-28 max-w-[45rem] text-center leading-8 sm:mb-40 scroll-mt-28"
      initial={{ opacity: 0, y: 100 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.175 }}
      id="about"
    >
      <SectionHeading>About me</SectionHeading>
      <p className="mb-3">
        I'm a student at the University of Maryland, passionate about leveraging{" "}
        <span className="font-medium">AI and software engineering</span> to create innovative solutions.
        My journey in tech is driven by a curiosity to explore the boundless possibilities of{" "}
        <span className="font-medium">machine learning</span> and entrepreneurship.
        Currently, I'm honing my skills as a{" "}
        <span className="font-medium">Software Engineering Intern at Halvik</span>, focusing on{" "}
        <span className="font-medium">Generative AI applications</span>.
      </p>

      <p>
        <span className="italic">When I'm not coding</span>, you can find me weightlifting, playing basketball,
        experimenting with new recipes in the kitchen, or spending time with my dog.
        I'm always looking to learn new technologies and improve my skills.
      </p>
    </motion.section>
  );
}