import React from "react";
import { CgWorkAlt } from "react-icons/cg";
import { FaReact } from "react-icons/fa";
import { LuGraduationCap } from "react-icons/lu";
import nbaImg from "@/public/nba.png";
import skycastImg from "@/public/skycast.png";
import marylandImg from "@/public/maryland.png";

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
    location: "Vienna, VA",
    description:
      "Developed a Generative AI application tailored for document retrieval and proposal generation by implementing a private Retrieval-Augmented Generation (RAG) system using AWS Bedrock, which improved document retrieval accuracy by 25% and increased proposal acceptance rate by 17%. Implemented efficient processing and management solutions using AWS S3 and Lambda, reducing latency by 11%. Leveraged Jupyter Notebooks for iterative development and testing within an Agile workflow, and utilized Docker for containerization to ensure consistent deployment. Streamlined the CI/CD pipeline design and implementation using AWS KMS and IAM, balancing robust security measures with ease of use.",
    icon: React.createElement(CgWorkAlt),
    date: "May 2024 – Present",
  },
  {
    title: "Product Innovation Engineer",
    location: "College Park, MD",
    description:
      "1 of 10 students selected for a prestigious 15-month program focused on efforts to improve school safety. Leading the development of an MVP, fine-tuning LLMs with a curated dataset for analyzing textual threats, and integrating TensorFlow for computer vision, resulting in the detection and notification of key personnel about active shooter threats on campuses, backed by a $250K-$2M investment.",
    icon: React.createElement(CgWorkAlt),
    date: "Jan. 2024 – Present",
  },
  {
    title: "Facial Recognition Researcher",
    location: "College Park, MD",
    description:
      "Utilized R to analyze facial recognition data using regression and factor analysis, uncovering significant patterns that improved prediction accuracy by 33%. Identified an 8% accuracy variance in facial recognition across races, genders, and ages, and implemented enhanced algorithmic weighting and adjusted training data sets, reducing bias and improving overall accuracy.",
    icon: React.createElement(CgWorkAlt),
    date: "Aug. 2023 – Present",
  },
] as const;

export const projectsData = [
  {
    title: "NBA Game Predictor",
    description:
      "Built a model to predict game outcomes using player and team data, achieving high accuracy.",
    tags: ["Python", "Pandas", "Scikit-Learn"],
    imageUrl: nbaImg,
  },
  {
    title: "SkyCast: Flight Delay Predictor",
    description:
      "Developed an app to predict flight delays using real-time weather data, with a mean absolute error of 1.48 minutes.",
    tags: ["Flask", "React", "Pandas", "Scikit-Learn"],
    imageUrl: skycastImg,
  },
  {
    title: "University Registration System",
    description:
      "Built a scalable course registration system with improved performance and reliability.",
    tags: ["Java", "Multi-threading"],
    imageUrl: marylandImg,
  },
] as const;

export const skillsData = [
  "Python",
  "Java",
  "C",
  "C++",
  "JavaScript",
  "MATLAB",
  "HTML/CSS",
  "R",
  "SQL",
  "Assembly",
  "React.js",
  "Node.js",
  "Pandas",
  "Scikit-Learn",
  "Flask",
  "OpenCV",
  "TensorFlow",
  "AWS Services",
  "Docker",
  "LLMs",
  "MySQL",
  "Jupyter",
  "Jira",
  "MongoDB",
  "UNIX/Linux",
  "Git",
] as const;
