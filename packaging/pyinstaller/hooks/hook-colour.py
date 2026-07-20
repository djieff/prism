"""PyInstaller hook for Prism's colour-science usage.

Prism uses colour-science for numeric colour data, not for Matplotlib plotting.
Colour imports its own plotting package during initialisation, so keep
``colour.plotting`` available and exclude only external plotting dependencies.
"""

excludedimports = [
    "matplotlib",
    "matplotlib.axes",
    "matplotlib.cm",
    "matplotlib.collections",
    "matplotlib.colors",
    "matplotlib.figure",
    "matplotlib.font_manager",
    "matplotlib.patches",
    "matplotlib.path",
    "matplotlib.pyplot",
    "matplotlib.ticker",
    "mpl_toolkits",
    "mpl_toolkits.mplot3d",
]
