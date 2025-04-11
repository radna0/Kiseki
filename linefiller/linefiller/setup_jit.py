from setuptools import setup, Extension
import pybind11
import subprocess

# Get OpenCV compile and link flags from pkg-config.
opencv_cflags = (
    subprocess.check_output("pkg-config --cflags opencv4", shell=True)
    .decode()
    .strip()
    .split()
)
opencv_libs = (
    subprocess.check_output("pkg-config --libs opencv4", shell=True)
    .decode()
    .strip()
    .split()
)

setup(
    ext_modules=[
        Extension(
            "trappedballcpp",
            ["trappedballcpp.cpp"],
            include_dirs=[
                pybind11.get_include(),
                pybind11.get_include(user=True),
                "/usr/include/opencv4",
            ],
            language="c++",
            extra_compile_args=["-std=c++17", "-O3", "-march=native", "-flto"]
            + opencv_cflags,
            extra_link_args=opencv_libs,
        ),
    ],
)
