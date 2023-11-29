from setuptools import setup, Extension

setup(
    name="gcyc",
    version="0.1",
    description="A simple directed graph cycle detection",
    ext_modules=[Extension("gcyc", sources=["gcycmodule.c"], py_limited_api=True)],
    install_requires=[],
)
