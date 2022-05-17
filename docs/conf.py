import os
import sys
from datetime import date

# Adds path to local extension
sys.path.insert(0, os.path.abspath('../archai'))

# Project information
project = 'Archai'
author = 'Microsoft'
copyright = f'{date.today().year}, {author}.'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
    'sphinx.ext.viewcode',
    'sphinxawesome_theme',
    'myst_parser',
    'sphinx_sitemap',
    'sphinxcontrib.programoutput',
]

myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'replacements',
    'substitution'
]

exclude_patterns = [
    'confs/**',
    'dockers/**',
    'models/**',
    'scripts/**',
    'tests/**',
    'tools/**',
]

# Options for HTML output
html_title = project
html_theme = 'sphinxawesome_theme'
html_logo = 'assets/img/logo.png'
html_favicon = 'assets/img/favicon.ico'

html_last_updated_fmt = ''
html_static_path = ['assets']
html_css_files = ['css/custom.css']

html_permalinks_icon = (
    '<svg xmlns="http://www.w3.org/2000/svg" '
    'viewBox="0 0 24 24">'
    '<path d="M3.9 12c0-1.71 1.39-3.1 '
    "3.1-3.1h4V7H7c-2.76 0-5 2.24-5 5s2.24 "
    "5 5 5h4v-1.9H7c-1.71 0-3.1-1.39-3.1-3.1zM8 "
    "13h8v-2H8v2zm9-6h-4v1.9h4c1.71 0 3.1 1.39 3.1 "
    "3.1s-1.39 3.1-3.1 3.1h-4V17h4c2.76 0 5-2.24 "
    '5-5s-2.24-5-5-5z"/></svg>'
)

html_theme_options = {
    'show_scrolltop': True,
    'extra_header_links': {
        'repository on GitHub.com': {
            'link': 'https://github.com/microsoft/archai',
            'icon': (
                '<svg style="height: 26px; margin-top: -2px;" viewBox="0 0 45 44" '
                'fill="currentColor" xmlns="http://www.w3.org/2000/svg">'
                '<path fill-rule="evenodd" clip-rule="evenodd" '
                'd="M22.477.927C10.485.927.76 10.65.76 22.647c0 9.596 6.223 17.736 '
                '14.853 20.608 1.087.2 1.483-.47 1.483-1.047 '
                '0-.516-.019-1.881-.03-3.693-6.04 '
                '1.312-7.315-2.912-7.315-2.912-.988-2.51-2.412-3.178-2.412-3.178-1.972-1.346.149-1.32.149-1.32 '  # noqa
                '2.18.154 3.327 2.24 3.327 2.24 1.937 3.318 5.084 2.36 6.321 '
                '1.803.197-1.403.759-2.36 '
                '1.379-2.903-4.823-.548-9.894-2.412-9.894-10.734 '
                '0-2.37.847-4.31 2.236-5.828-.224-.55-.969-2.759.214-5.748 0 0 '
                '1.822-.584 5.972 2.226 '
                '1.732-.482 3.59-.722 5.437-.732 1.845.01 3.703.25 5.437.732 '
                '4.147-2.81 5.967-2.226 '
                '5.967-2.226 1.185 2.99.44 5.198.217 5.748 1.392 1.517 2.232 3.457 '
                '2.232 5.828 0 '
                '8.344-5.078 10.18-9.916 10.717.779.67 1.474 1.996 1.474 4.021 0 '
                '2.904-.027 5.247-.027 '
                '5.96 0 .58.392 1.256 1.493 1.044C37.981 40.375 44.2 32.24 44.2 '
                '22.647c0-11.996-9.726-21.72-21.722-21.72" '
                'fill="currentColor"/></svg>'
            )
        },
        'Microsoft.com': {
            'link': 'https://microsoft.com',
            'icon': (
                '<svg style="height: 26px; margin-top: -2px;" viewBox="0 0 23 23" '
                'fill="currentColor" xmlns="http://www.w3.org/2000/svg">'
                '<path fill="#f35325" d="M1 1h10v10H1z"/>'
                '<path fill="#81bc06" d="M12 1h10v10H12z"/>'
                '<path fill="#05a6f0" d="M1 12h10v10H1z"/>'
                '<path fill="#ffba08" d="M12 12h10v10H12z"/>'
                '</svg>'
            )
        }
    }
}
