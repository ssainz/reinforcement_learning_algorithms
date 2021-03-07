import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(

    name="reinforcement_learning_algorithms", # Replace with your username

    version="1.0.0",

    author="ssainz",

    author_email="<sergio.sainz@gmail.com>",

    description="reinforcement_learning_algorithms",

    long_description=long_description,

    long_description_content_type="text/markdown",

    url="<https://github.com/ssainz/reinforcement_learning_algorithms>",

    packages=setuptools.find_packages(),

    classifiers=[

        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",

    ],

    python_requires='>=3.6',

    install_requires=['torch',
                      'gym',
                      'torchvision',
                      'numpy',
                      ],

)