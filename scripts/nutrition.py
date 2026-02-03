import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from nutri_plots import *

def main():
    
    img_root = "imgs/nutrition/"

    if (not os.path.exists(img_root)):
        os.makedirs(img_root)
    
    # radial_view = food_groups_radial_view()
    # radial_view.savefig(f"{img_root}/food_groups_radial_view.png")
    
    hdata = pd.read_csv("OUTCOME_DIAGNOSIS_processed.csv", header=0)
    
    # boxplot_z_score_multi = boxplot_z_score_foods_per_multi_group(hdata)
    # boxplot_z_score_multi.savefig(f"{img_root}/boxplot_z_score_foods_per_multi_group.png")

    boxplot_z_score_single = boxplot_z_score_foods_per_single_group(hdata)
    boxplot_z_score_single.savefig(f"{img_root}/boxplot_z_score_foods_per_single_group.png")
    
if __name__ == "__main__":
    main()