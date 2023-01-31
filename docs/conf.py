# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
from datetime import date

# Adds path to local extension
sys.path.insert(0, os.path.abspath(".."))

# Project information
project = "Archai"
author = "Microsoft"
copyright = f"{date.today().year}"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_sitemap",
    "sphinxcontrib.programoutput",
    "sphinxcontrib.mermaid",
    "sphinx_inline_tabs",
    "sphinx_git",
    "nbsphinx",
]
exclude_patterns = [
    "benchmarks/**",
    "devices/**",
    "devops/**",
    "docker/**",
    "scripts/**",
    "tests/**",
]
extlinks = {"github": ("https://github.com/microsoft/archai/tree/main/%s", "%s")}
source_suffix = ".rst"
master_doc = "index"
language = "en"

# Options for HTML output
html_title = project
html_theme = "sphinx_book_theme"
html_logo = "assets/img/logo.png"
html_favicon = "assets/img/favicon.ico"
html_last_updated_fmt = ""
html_static_path = ["assets"]
html_css_files = ["css/custom.css"]
html_theme_options = {
    "repository_url": "https://github.com/microsoft/archai",
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": False,
    "use_fullscreen_button": False,
    "use_repository_button": True,
    "logo_only": True,
    "show_navbar_depth": 1,
    "toc_title": "Sections",
}

# Autodoc
autodoc_default_options = {"exclude-members": "__weakref__"}
autodoc_member_order = "bysource"
autodoc_mock_imports = ["ray.tune", "xautodl"]

# Disables `nbsphinx` require.js to avoid
# conflicts with `sphinxcontrib.mermaid`
nbsphinx_requirejs_path = ""
