# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#
import os
import sys

home_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, home_path)

# -- General configuration ------------------------------------------------

extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.ifconfig',
    'sphinx_design',
    'reno.sphinxext',
]

autosummary_generate = True

source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

language = 'en'

add_module_names = False

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'sdk_index.rst']

linkcheck_retries = 2
linkcheck_anchors = False
linkcheck_ignore = [r'https://cloud.dwavesys.com/leap',  # redirects, many checks
                    ]
# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

doctest_global_setup = \
    """
    """

# -- Options for HTML output ----------------------------------------------

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "collapse_navigation": True,
    "show_prev_next": False,
}
html_sidebars = {"**": ["search-field", "sidebar-nav-bs"]}  # remove ads

# TODO: verify the link to dwave docs
intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
    'qbsolv': ('https://docs.ocean.dwavesys.com/projects/qbsolv/en/latest/', None),
    'dwave': ('https://docs.dwavequantum.com/en/latest/', None),
    }
