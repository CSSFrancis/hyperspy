# -*- coding: utf-8 -*-
#
# HyperSpy documentation build configuration file, created by
# sphinx-quickstart on Mon Oct 18 11:10:55 2010.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import sys
from datetime import datetime
from importlib.metadata import version as get_version

import hyperspy

sys.path.append("../")

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath('.'))

# -- General configuration -----------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "IPython.sphinxext.ipython_directive",  # Needed in basic_usage.rst
    "numpydoc",
    "sphinxcontrib.towncrier",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.graphviz",
    "sphinx.ext.mathjax",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery",
    "sphinx_copybutton",
    "sphinx_favicon",
]

linkcheck_ignore = [
    "https://anaconda.org",  # 403 Client Error: Forbidden for url
    "https://doi.org/10.1021/acs.nanolett.5b00449",  # 403 Client Error: Forbidden for url
    "https://onlinelibrary.wiley.com",  # 403 Client Error: Forbidden for url
    "https://www.jstor.org/stable/24307705",  # 403 Client Error: Forbidden for url
    "https://software.opensuse.org",  # 400 Client Error: Bad Request for url
]

linkcheck_exclude_documents = []

# Specify a standard user agent, as Sphinx default is blocked on some sites
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 Edg/108.0.1462.54"

try:
    import sphinxcontrib.spelling  # noqa: F401

    extensions.append("sphinxcontrib.spelling")
except BaseException:
    pass
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

autosummary_generate = True

# The suffix of source filenames.
source_suffix = ".rst"

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "HyperSpy"
copyright = f"2011-{datetime.today().year}, The HyperSpy development team"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The full version, including alpha/beta/rc tags.
release = get_version("hyperspy")
# The short X.Y version.
version = ".".join(release.split(".")[:2])

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
# language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
# today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build"]

# The reST default role (used for this markup: `text`) to use for all documents.
# default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []


# -- Options for HTML output ---------------------------------------------

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# html_theme_options = {'collapsiblesidebar': True}

# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
# html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/hyperspy_logo.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

favicons = [
    "hyperspy.ico",
]

# For version switcher:
# For development, we match to the dev version in `switcher.json`
# for release version, we match to the minor increment
_version = hyperspy.__version__
version_match = "dev" if "dev" in _version else ".".join(_version.split(".")[:2])

print("version_match:", version_match)

html_theme_options = {
    "analytics": {
        "google_analytics_id": "G-B0XD0GTW1M",
    },
    "show_toc_level": 2,
    "github_url": "https://github.com/hyperspy/hyperspy",
    "icon_links": [
        {
            "name": "Gitter",
            "url": "https://gitter.im/hyperspy/hyperspy",
            "icon": "fab fa-gitter",
        },
        {
            "name": "HyperSpy",
            "url": "https://hyperspy.org",
            "icon": "_static/hyperspy.ico",
            "type": "local",
        },
    ],
    "logo": {
        "text": "HyperSpy",
    },
    "external_links": [
        {
            "url": "https://github.com/hyperspy/hyperspy-demos",
            "name": "Tutorial",
        },
    ],
    "header_links_before_dropdown": 7,
    "switcher": {
        # Update when merged and released
        "json_url": "https://hyperspy.org/hyperspy-doc/dev/_static/switcher.json",
        "version_match": version_match,
    },
    "navbar_start": ["navbar-logo", "version-switcher"],
    "announcement": "HyperSpy API is changing in version 2.0, see the <a href='https://hyperspy.org/hyperspy-doc/current/changes.html'>release notes!</a>",
}
# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
# html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
# html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If false, no module index is generated.
# html_domain_indices = True

# If false, no index is generated.
# html_use_index = True

# If true, the index is split into individual pages for each letter.
# html_split_index = False

# If true, links to the reST sources are added to the pages.
# html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
# html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
# html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = "HyperSpydoc"

# Add the documentation for __init__() methods and the class docstring to the
# built documentation
autoclass_content = "both"

# -- Options for LaTeX output --------------------------------------------

# The paper size ('letter' or 'a4').
# latex_paper_size = 'letter'

# The font size ('10pt', '11pt' or '12pt').
# latex_font_size = '10pt'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
    (
        "index",
        "HyperSpy.tex",
        "HyperSpy Documentation",
        "The HyperSpy Developers",
        "manual",
    ),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Additional stuff for the LaTeX preamble.
# latex_preamble = ''

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_domain_indices = True

# -- Options for manual page output --------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ("index", "hyperspy", "HyperSpy Documentation", ["The HyperSpy developers"], 1)
]


# -- Options for towncrier_draft extension -----------------------------------

# Options: draft/sphinx-version/sphinx-release
towncrier_draft_autoversion_mode = "draft"
towncrier_draft_include_empty = False
towncrier_draft_working_directory = ".."

# Add the hyperspy website to the intersphinx domains
intersphinx_mapping = {
    "cupy": ("https://docs.cupy.dev/en/stable", None),
    "dask": ("https://docs.dask.org/en/latest", None),
    "exspy": ("https://exspy.readthedocs.io/en/latest", None),
    "h5py": ("https://docs.h5py.org/en/stable", None),
    "holospy": ("https://holospy.readthedocs.io/en/latest", None),
    "IPython": ("https://ipython.readthedocs.io/en/stable", None),
    "ipyparallel": ("https://ipyparallel.readthedocs.io/en/latest", None),
    "mdp": ("https://mdp-toolkit.github.io/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pint": ("https://pint.readthedocs.io/en/stable", None),
    "python": ("https://docs.python.org/3", None),
    "rsciio": ("https://hyperspy.org/rosettasciio/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "skimage": ("https://scikit-image.org/docs/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "traits": ("https://docs.enthought.com/traits/", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable", None),
}

# Check links to API when building documentation
nitpicky = True
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-nitpick_ignore
nitpick_ignore_regex = (
    # No need to be added to the API: documented in subclass
    ("py:class", "hyperspy.misc.slicing.FancySlicing"),
    ("py:class", "hyperspy.learn.mva.MVA"),
    ("py:class", "hyperspy.signal.MVATools"),
    ("py:class", "hyperspy.samfire_utils.strategy.SamfireStrategy"),
    ("py:class", ".*goodness_test"),
    ("py:class", "hyperspy.roi.BasePointROI"),
    # Add exception to API
    ("py:obj", "SignalDimensionError"),
    ("py:obj", "DataDimensionError"),
    # Need to be made a property
    ("py:attr", "api.signals.BaseSignal.learning_results"),
    ("py:attr", "api.signals.BaseSignal.axes_manager"),
    ("py:attr", "hyperspy._signals.lazy.LazySignal.navigator"),
    # Skip for now
    ("py:attr", "axes.BaseDataAxis.is_binned.*"),
    ("py:attr", "api.model.components1D.ScalableFixedPattern.*"),
    ("py:class", ".*HistogramTilePlot"),
    ("py:class", ".*SquareCollection"),
    ("py:class", ".*RectangleCollection"),
    # Traits property not playing well
    ("py:attr", "component.Parameter.*"),
    # Adding to the API reference not useful
    ("py:.*", "api.preferences.*"),
    # Unknown
    ("py:.*", "has_pool"),
)

# -- Options for numpydoc extension -----------------------------------

numpydoc_show_class_members = False
numpydoc_xref_param_type = True
numpydoc_xref_ignore = {
    "type",
    "optional",
    "default",
    "of",
    "or",
    "auto",
    "from_elements",
    "all_alpha",
    "subclass",
    "dask",
    "scheduler",
    "matplotlib",
    "color",
    "line",
    "style",
    "hyperspy",
    "widget",
    "strategy",
    "module",
}

# if Version(numpydoc.__version__) >= Version("1.6.0rc0"):
#     numpydoc_validation_checks = {"all", "ES01", "EX01", "GL02", "GL03", "SA01", "SS06"}

autoclass_content = "both"

autodoc_default_options = {
    "show-inheritance": True,
}
toc_object_entries_show_parents = "hide"
numpydoc_class_members_toctree = False

# -- Sphinx-Gallery---------------

# https://sphinx-gallery.github.io
sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # path to your example scripts
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
    "filename_pattern": ".py",  # pattern to define which will be executed
    "ignore_pattern": "_sgskip.py",  # pattern to define which will not be executed
}

graphviz_output_format = "svg"

# -- Sphinx-copybutton -----------


copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True


def setup(app):
    app.add_css_file("custom-styles.css")


tls_verify = False
