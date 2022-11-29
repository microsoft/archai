# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
from datetime import date

# Adds path to local extension
sys.path.insert(0, os.path.abspath("../archai"))

# Project information
project = "Archai"
author = "Microsoft"
copyright = f"{date.today().year}, {author}"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_sitemap",
    "sphinxcontrib.programoutput",
]

myst_enable_extensions = ["colon_fence", "deflist", "replacements", "substitution"]

exclude_patterns = [
    "benchmarks/**",
    "devices/**",
    "devops/**",
    "docker/**",
    "scripts/**",
    "tests/**",
]

autodoc_member_order = "bysource"

# Options for HTML output
html_title = project
html_theme = "sphinx_rtd_theme"
html_logo = "assets/img/logo.png"
html_favicon = "assets/img/favicon.ico"

html_last_updated_fmt = ""
html_static_path = ["assets"]
html_css_files = ["css/custom.css"]

html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": True,
}
