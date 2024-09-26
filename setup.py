from setuptools import setup, find_packages

setup(
    name="Qwixx",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'setuptools',
        'tqdm',
    ],
    author="Ignace Konig",
    author_email="sappie.konig@gmail.com",
    description="A simple interface for the qwixx game for reinforcement learning purposes",
    url="https://github.com/SappieKonig/qwixx",
)
