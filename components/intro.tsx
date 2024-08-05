import React from 'react';
import Image from 'next/image';
import { motion } from 'framer-motion';

export default function About() {
    return (
        <section id="about" className="py-20 bg-gray-100">
            <div className="container mx-auto px-4">
                <motion.h1
                    className="text-4xl font-bold text-center mb-12"
                    initial={{ opacity: 0, scale: 0 }}
                    whileInView={{ opacity: 1, scale: 1 }}
                >
                    About
                </motion.h1>
                <div className="flex flex-col md:flex-row items-center gap-8">
                    <motion.div
                        className="md:w-1/2"
                        initial={{ opacity: 0 }}
                        whileInView={{ opacity: 1 }}
                    >
                        <Image
                            src="/tahoe.jpg"
                            alt="Tahoe"
                            width={500}
                            height={300}
                            objectFit="cover"
                            className="rounded-lg shadow-lg"
                        />
                    </motion.div>
                    <motion.div
                        className="md:w-1/2 bg-white p-8 rounded-lg shadow-lg"
                        initial={{ opacity: 0 }}
                        whileInView={{ opacity: 1 }}
                    >
                        <p className="mb-4">
                            I'm a student at the University of Maryland, passionate about leveraging AI and software engineering to create innovative solutions. My journey in tech is driven by a curiosity to explore the boundless possibilities of machine learning and entrepreneurship.
                        </p>
                        <p className="mb-4">
                            Currently, I'm honing my skills as a Software Engineering Intern at Halvik, focusing on Generative AI applications.
                        </p>
                        <p>
                            When I'm not coding, you can find me weightlifting, playing basketball, experimenting with new recipes in the kitchen, or spending time with my dog. I'm always looking to learn new technologies and improve my skills.
                        </p>
                    </motion.div>
                </div>
            </div>
        </section>
    );
}