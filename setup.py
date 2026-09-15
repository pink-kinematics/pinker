# SPDX-License-Identifier: Apache-2.0

"""Build the pinker.kinematics C extension."""

import numpy
from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "pinker.kinematics._kinematics_c",
            sources=["pinker/kinematics/_kinematics_c.c"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-O2", "-std=c99"],
        )
    ]
)
