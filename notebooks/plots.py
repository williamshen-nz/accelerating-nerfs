def load_mpl_style():
    """Load a nicer matplotlib style"""
    import matplotlib as mpl
    import matplotlib.font_manager as font_manager

    mpl.rcParams["font.family"] = "serif"
    cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + "/fonts/ttf/cmr10.ttf")
    mpl.rcParams["font.serif"] = cmfont.get_name()
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams["axes.unicode_minus"] = False
    # mpl complains otherwise
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"
    mpl.rcParams["axes.formatter.use_mathtext"] = True
    # set fontsize
    mpl.rcParams["xtick.labelsize"] = 12
    mpl.rcParams["ytick.labelsize"] = 12
    mpl.rcParams["axes.labelsize"] = 12
    mpl.rcParams["legend.fontsize"] = 12
    mpl.rcParams["axes.titlesize"] = 14
