# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
from datetime import date

import sphinx_rtd_theme

# Adds path to local extension
sys.path.insert(0, os.path.abspath(".."))

# Project information
project = "Archai"
author = "Microsoft"
copyright = f"{date.today().year}, {author}"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
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
source_suffix = ".rst"
master_doc = "index"
language = "en"
pygments_style = "sphinx"

# Options for HTML output
html_title = project
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_logo = "assets/img/logo.png"
html_favicon = "assets/img/favicon.ico"
html_last_updated_fmt = ""
html_static_path = ["assets"]
html_css_files = ["css/custom.css"]
html_theme_options = {
    "collapse_navigation": False,
    "display_version": False,
    "logo_only": True,
    "navigation_depth": 4,
}
html_context = {"display_github": True}

# Autodoc
autodoc_default_options = {"exclude-members": "__weakref__"}
autodoc_member_order = "bysource"