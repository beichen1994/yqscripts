#!/usr/bin/env python
#-*- coding:utf-8 -*-



from setuptools import setup, find_packages

setup(
    name = "yqscripts",
    version = "0.1.0",
    keywords = ("yq scripts"),
    description = "my useful scripts",
    long_description = "my useful scripts",
    license = "MIT Licence",

    url = "https://github.com/beichen1994/yqscripts",
    author = "Yuqi Wang",
    author_email = "1768229565@qq.com",

    packages = find_packages(),
    include_package_data = True,
    platforms = "any",
    install_requires = ['requests']
)
