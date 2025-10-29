import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

project = 'hessband'
copyright = '2024, Gaurav Sood'
author = 'Gaurav Sood'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'furo'
html_static_path = ['_static']

# Furo theme options
html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#2563eb",
        "color-brand-content": "#2563eb",
    },
    "dark_css_variables": {
        "color-brand-primary": "#60a5fa",
        "color-brand-content": "#60a5fa",
    },
}

autodoc_member_order = 'bysource'
napoleon_google_docstring = True
napoleon_numpy_docstring = True