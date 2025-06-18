from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.1.0"

ext_modules = [
    Pybind11Extension(
        "carmack_core",
        ["carmack_core_simple.cpp"],
        include_dirs=[
            "/usr/local/include",
            "/usr/include/opencv4",
        ],
        libraries=["opencv_core", "opencv_imgproc", "gomp"],
        library_dirs=["/usr/local/lib", "/usr/lib/x86_64-linux-gnu"],
        language='c++',
        cxx_std=17,
        extra_compile_args=[
            "-O3",
            "-march=native",
            "-fopenmp",
            "-ffast-math",
            "-funroll-loops",
            "-ftree-vectorize",
            "-DNDEBUG",
        ],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="carmack_core",
    version=__version__,
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)