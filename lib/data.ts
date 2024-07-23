import React from "react";
import { CgWorkAlt } from "react-icons/cg";
import { FaReact } from "react-icons/fa";
import { LuGraduationCap } from "react-icons/lu";
import corpcommentImg from "@/public/corpcomment.png";
import rmtdevImg from "@/public/rmtdev.png";
import wordanalyticsImg from "@/public/wordanalytics.png";

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
      "Built a machine learning model with advanced feature engineering, data cleaning, and dimensional reduction. Collected and processed player-level and team-level data from open-source APIs, creating features representing team averages over the previous 5, 10, and 15 games. Implemented dimensionality reduction using Principal Component Analysis (PCA) and trained models including Random Forest and Logistic Regression, achieving high prediction accuracy and generating insightful evaluation.",
    tags: ["Python", "Pandas", "Scikit-Learn"],
    imageUrl: corpcommentImg,
  },
  {
    title: "SkyCast: Flight Delay Predictor",
    description:
      "Developed a Flask-based application that harnesses the Open-Meteo API to predict flight delays by analyzing 5 real-time weather metrics: temperature, precipitation, wind speed, pressure, and visibility. Engineered a RandomForestRegressor model, achieving a mean absolute error (MAE) of 1.48 minutes. Designed a scalable React interface for dynamic user interaction; integrated with a Flask backend to handle over 33,000 API calls daily, ensuring application robustness and efficient traffic management.",
    tags: ["Flask", "React", "Pandas", "Scikit-Learn"],
    imageUrl: rmtdevImg,
  },
  {
    title: "University Registration System",
    description:
      "Built a scalable system for course registration, allowing addition and cancellation of courses. Implemented sets and multi-threading to improve performance and developed error handling for reliable operation.",
    tags: ["Java", "Multi-threading"],
    imageUrl: wordanalyticsImg,
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
