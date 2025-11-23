# Configuration file for the Sphinx documentation builder.

from importlib import metadata

project = metadata.metadata("hessband")["Name"]
# Extract author name from authors list in pyproject.toml
author_info = metadata.metadata("hessband").get("Author-email", "")
if author_info:
    # Parse "Name <email>" format
    author = author_info.split(" <")[0] if " <" in author_info else author_info
else:
    author = "Gaurav Sood"  # fallback
release = metadata.version("hessband")

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = f"{project} v{release}"
html_static_path = ["_static"]

# -- Extension configuration -------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
