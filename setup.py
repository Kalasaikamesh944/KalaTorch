from setuptools import setup, find_packages

setup(
    name="kalatorch",
    version="0.0.0",
    description="A high-level PyTorch framework for easy implementation of AI models and neural networks.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="N V R K SAI KAMESH YADAVALLI",
    author_email="your_email@example.com",
    url="https://github.com/Kalasaikamesh944/KalaTorch.git",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "torchvision>=0.9.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    license="MIT",
    include_package_data=True,
    keywords="pytorch ai machine-learning deep-learning neural-networks",
)
