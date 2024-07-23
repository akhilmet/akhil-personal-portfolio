import React from "react";
import { CgWorkAlt } from "react-icons/cg";
import { FaReact, FaAws, FaJava } from "react-icons/fa";
import { LuGraduationCap } from "react-icons/lu";
import { SiPython, SiTensorflow, SiR, SiFlask, SiPandas, SiScikitlearn, SiJavascript, SiCplusplus, SiC, SiSql, SiNodeDotJs, SiNextDotJs, SiAmazonaws, SiGithub, SiDocker, SiGit, SiJupyter, SiJira, SiMongodb, SiUnix, SiLinux } from "react-icons/si";
import { IoSchool } from "react-icons/io5";

import nbapredictorImg from "@/public/nbapredictor.png";
import skycastImg from "@/public/skycast.png";
import registrationImg from "@/public/registration.png";

export const links = [
  {
    name: "Home",
    hash: "#home",
  },
  {
    name: "About",
    hash: "#about",
  },
  {
    name: "Projects",
    hash: "#projects",
  },
  {
    name: "Skills",
    hash: "#skills",
  },
  {
    name: "Experience",
    hash: "#experience",
  },
  {
    name: "Contact",
    hash: "#contact",
  },
] as const;

export const experiencesData = [
  {
    title: "Software Engineering Intern - Generative AI",
    location: "Halvik, Vienna, VA",
    description:
      "Developed a Generative AI application tailored for document retrieval and proposal generation using AWS Bedrock. Improved document retrieval accuracy by 25% and increased proposal acceptance rate by 17%. Implemented efficient processing and management solutions using AWS S3 and Lambda, reducing latency by 11%.",
    icon: React.createElement(FaAws),
    date: "May 2024 – Present",
  },
  {
    title: "Product Innovation Engineer",
    location: "xFoundry@UMD, College Park, MD",
    description:
      "Leading MVP development and fine-tuning LLMs for analyzing textual threats. Integrating TensorFlow for computer vision to detect and notify about active threats on campuses.",
    icon: React.createElement(SiTensorflow),
    date: "Jan. 2024 – Present",
  },
  {
    title: "Facial Recognition Researcher",
    location: "UMD FIRE, College Park, MD",
    description:
      "Analyzed facial recognition data using R, improving prediction accuracy by 33%. Identified an 8% accuracy variance across demographics, implementing solutions to reduce bias.",
    icon: React.createElement(SiR),
    date: "Aug. 2023 – Present",
  },
] as const;

export const projectsData = [
  {
    title: "NBA Game Predictor",
    description:
      "Machine learning model for predicting NBA game outcomes with advanced feature engineering and dimensionality reduction techniques.",
    tags: ["Python", "Pandas", "Scikit-Learn"],
    imageUrl: nbapredictorImg,
  },
  {
    title: "SkyCast: Flight Delay Predictor",
    description:
      "Flask-based application predicting flight delays using real-time weather data and machine learning, featuring a scalable React interface.",
    tags: ["Flask", "React", "Pandas", "Scikit-Learn"],
    imageUrl: skycastImg,
  },
  {
    title: "University Registration System",
    description:
      "Scalable system for course registration, implementing multi-threading and error handling for improved performance and reliability.",
    tags: ["Java"],
    imageUrl: registrationImg,
  },
] as const;

export const skillsData = [
  { skill: "Python", icon: SiPython },
  { skill: "Java", icon: FaJava },
  { skill: "C", icon: SiC },
  { skill: "C++", icon: SiCplusplus },
  { skill: "JavaScript", icon: SiJavascript },
  { skill: "SQL", icon: SiSql },
  { skill: "React", icon: FaReact },
  { skill: "Node.js", icon: SiNodeDotJs },
  { skill: "Next.js", icon: SiNextDotJs },
  { skill: "Flask", icon: SiFlask },
  { skill: "Pandas", icon: SiPandas },
  { skill: "Scikit-Learn", icon: SiScikitlearn },
  { skill: "AWS", icon: SiAmazonaws },
  { skill: "Git", icon: SiGit },
  { skill: "Docker", icon: SiDocker },
  { skill: "Jupyter", icon: SiJupyter },
  { skill: "Jira", icon: SiJira },
  { skill: "MongoDB", icon: SiMongodb },
  { skill: "UNIX", icon: SiUnix },
  { skill: "Linux", icon: SiLinux },
] as const;
