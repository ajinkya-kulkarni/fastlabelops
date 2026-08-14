import sys

import numpy
from setuptools import Extension, setup

if sys.platform == "win32":
    compile_args = ["/O2", "/DNDEBUG", "/std:c++17"]
else:
    compile_args = ["-O3", "-DNDEBUG", "-std=c++17"]

_CORE_INC = [
    "src/fastlabelops/_bindings.inc",
    "src/fastlabelops/_label_map.inc",
    "src/fastlabelops/_counts.inc",
    "src/fastlabelops/_relabel.inc",
    "src/fastlabelops/_remove_small.inc",
    "src/fastlabelops/_overlap.inc",
    "src/fastlabelops/_props.inc",
]

ext_modules = [
    Extension(
        "fastlabelops._core",
        ["src/fastlabelops/_core.cpp"],
        include_dirs=[numpy.get_include()],
        language="c++",
        extra_compile_args=compile_args,
        depends=_CORE_INC,
    )
]

setup(
    package_dir={"": "src"},
    packages=["fastlabelops"],
    package_data={"fastlabelops": ["*.pyi", "py.typed"]},
    ext_modules=ext_modules,
    zip_safe=False,
)
