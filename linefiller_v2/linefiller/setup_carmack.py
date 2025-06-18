from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import glob

__version__ = "0.1.0"

ext_modules = [
    Pybind11Extension(
        "carmack_core",
        ["carmack_core.cpp"],
        include_dirs=[
            "/usr/local/include",
            "/usr/include/opencv4",
        ],
        libraries=["opencv_core", "opencv_imgproc", "tbb", "gomp"],
        library_dirs=["/usr/local/lib", "/usr/lib/x86_64-linux-gnu"],
        language='c++',
        cxx_std=17,
        extra_compile_args=[
            "-O3",
            "-march=native",
            "-mavx512f",
            "-mavx512dq",
            "-mavx512cd",
            "-mavx512bw",
            "-mavx512vl",
            "-fopenmp",
            "-ffast-math",
            "-funroll-loops",
            "-ftree-vectorize",
        ],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="carmack_core",
    version=__version__,
    author="John Carmack (simulated)",
    author_email="carmack@idsoftware.com",
    url="https://github.com/id-Software/carmack-linefiller",
    description="God-tier performance line art colorization",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)