import matplotlib.pyplot as plt
import numpy as np
nutridict = {
    "FFQ1": "Milk_Yoghurt",
    "FFQ2": "Milk_Yoghurt_LowFat",
    "FFQ3": "Yellow_Cheese",
    "FFQ4": "Feta_Cheese",
    "FFQ5": "Cheese_LowFat",
    "FFQ6": "Eggs",
    "FFQ7": "White_Bread",
    "FFQ8": "Wholemeal_Bread",
    "FFQ11": "Cereals",
    "FFQ12": "White_Rice",
    "FFQ13": "Brown_Rice",
    "FFQ14": "Pasta",
    "FFQ15": "Pasta_Wholemeal",
    "FFQ16": "Potatoes_Boiled",
    "FFQ17": "Potatoes_Fried",
    "FFQ18": "Veal",
    "FFQ19": "Meat_Balls",
    "FFQ20": "Chicken",
    "FFQ21": "Pork",
    "FFQ22": "Lamb_Goat_Game",
    "FFQ23": "Meat_Cold_Sliced",
    "FFQ24": "Sausages",
    "FFQ25": "Meat_Cold_Sliced_LowFat",
    "FFQ26": "Fish_Small",
    "FFQ27": "Fish_Large",
    "FFQ28": "Seafood_Misc",
    "FFQ29": "Legumes",
    "FFQ30": "Spinach",
    "FFQ31": "Pastitsio_Moussaka_Papoutsakia",
    "FFQ32": "Petit_Pois",
    "FFQ33": "Tomato_Cucumber_Carrot_Pepper",
    "FFQ34": "Lettuce_Cabbage_Spinach_Rocket",
    "FFQ35": "Broccoli_Cauliflower_Courgette",
    "FFQ37": "Orange",
    "FFQ38": "Apples_Pears",
    "FFQ39": "Winter_Fruits_Misc",
    "FFQ40": "Banana",
    "FFQ41": "Summer_Fruits_Misc",
    "FFQ43": "Fruits_Dried",
    "FFQ44": "Nuts",
    "FFQ45": "Pies",
    "FFQ46": "Pies",
    "FFQ48": "Cakes",
    "FFQ49": "Sweets_Preserved",
    "FFQ50": "Cakes",
    "FFQ51": "Cakes",
    "FFQ52": "Chocolate",
    "FFQ53": "Ice_Cream",
    "FFQ54": "Salty_Snacks",
    "FFQ55": "Honey_Marmalade",
    "FFQ56": "Olives",
    "FFQ57": "Wine",
    "FFQ58": "Beer",
    "FFQ59": "Alcohold_Misc",
    "FFQ60": "Soda",
    "FFQ61": "Soda",
    "FFQ62": "Coffee",
    "FFQ63": "Tea",
    "FFQ64": "Mayonnaise",
    "FFQ65": "Mayonnaise",
    "FFQ66": "Olive_Oil_Freq",
    "FFQ67": "Seed_Oil_Freq",
    "FFQ68": "Margarine_Freq",
    "FFQ69": "Butter_Freq"
}

nutri_dict_grouped = {
    "Dairy": [
        "Milk_Yoghurt",
        "Milk_Yoghurt_LowFat",
        "Yellow_Cheese",
        "Feta_Cheese",
        "Cheese_LowFat"
    ],
    "Refined_Grains": [
        "White_Bread",
        "Cereals",
        "White_Rice",
        "Pasta"
    ],
    "Whole_Grains": [
        "Wholemeal_Bread",
        "Brown_Rice",
        "Pasta_Wholemeal"
    ],
    "Red_Meats": [
        "Veal",
        "Pork",
        "Lamb_Goat_Game"
    ],
    "Processed_Foods": [
        "Potatoes_Fried",
        "Meat_Balls",
        "Meat_Cold_Sliced",
        "Meat_Cold_Sliced_LowFat",
        "Salty_Snacks",
        "Mayonnaise",
        "Soda",
        "Sausages"
    ],
    "Seafood": [
        "Fish_Small",
        "Fish_Large",
        "Seafood_Misc"
    ],
    "Vegetables": [
        "Potatoes_Boiled",
        "Spinach",
        "Tomato_Cucumber_Carrot_Pepper",
        "Lettuce_Cabbage_Spinach_Rocket",
        "Broccoli_Cauliflower_Courgette"
    ],
    "Fruits": [
        "Orange",
        "Apples_Pears",
        "Winter_Fruits_Misc",
        "Banana",
        "Summer_Fruits_Misc"
    ],
    "Composite_Dishes": [
        "Pastitsio_Moussaka_Papoutsakia",
        "Pies"
    ],
    "Sweets": [
        "Cakes",
        "Sweets_Preserved",
        "Chocolate",
        "Ice_Cream",
        "Honey_Marmalade"
    ],
    "Alcohol": [
        "Wine",
        "Beer",
        "Alcohold_Misc"
    ]
}

single_items = {
    "Eggs": "Eggs",
    "Chicken": "White_Meats",
    "Legumes": "Legumes",
    "Petit_Pois": "Petit_Pois",
    "Fruits_Dried": "Fruits_Dried",
    "Nuts": "Nuts",
    "Olives": "Olives",
    "Coffee": "Coffee",
    "Tea": "Tea",
    "Olive_Oil_Freq": "Olive_Oil_Freq",
    "Seed_Oil_Freq": "Seed_Oil_Freq",
    "Margarine_Freq": "Margarine_Freq",
    "Butter_Freq": "Butter_Freq"
}

category_colors = {
    'Dairy': 'lightblue',
    'Refined_Grains': 'wheat',
    'Whole_Grains': 'lightcoral',
    'Red_Meats': 'lightgreen',
    'Processed_Foods': 'gold',
    'Seafood': 'pink',
    'Vegetables': 'plum',
    'Fruits': 'lavender',
    'Composite_Dishes': 'lightgray',
    'Sweets': 'purple',
    'Alcohol': 'red',
    'Eggs': '#FFFACD',
    'White_Meats': '#B0E0E6',
    'Legumes': '#A0522D',
    'Petit_Pois': '#32CD32',
    'Fruits_Dried': '#FFA07A',
    'Nuts': '#CD853F',
    'Olives': '#6B8E23',
    'Coffee': '#8B4513',
    'Tea': '#228B22',
    'Olive_Oil_Freq': '#DAA520',
    'Seed_Oil_Freq': '#FFD700',
    'Margarine_Freq': '#FFF8DC',
    'Butter_Freq': '#FFE4B5'
}

def food_groups_radial_view():
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

    group_names = list(nutri_dict_grouped.keys())
    n_groups = len(group_names)
    group_sizes = [len(foods) for foods in nutri_dict_grouped.values()]

    cmap = plt.cm.tab20c
    colors = [cmap(i/n_groups) for i in range(n_groups)]

    angles = np.linspace(0, 2*np.pi, n_groups, endpoint=False)
    width = 2*np.pi / n_groups

    bars = ax.bar(angles, [10]*n_groups, width=width, 
                bottom=5, color=colors, alpha=0.7,
                edgecolor='white', linewidth=2)

    for angle, group, size in zip(angles, group_names, group_sizes):
        rotation = np.degrees(angle + width/2)
        ha = 'left' if angle < np.pi else 'right'
        
        ax.text(angle + width/2, 12, f"{group}\n({size} items)",
                ha='center', va='center',
                rotation=rotation if rotation < 180 else rotation-180,
                fontsize=9, fontweight='bold')

    food_counts = []
    for group, foods in nutri_dict_grouped.items():
        for i, food in enumerate(foods):
            group_idx = list(nutri_dict_grouped.keys()).index(group)
            food_angle = angles[group_idx] + (i/len(foods)) * width
            
            ax.scatter(food_angle, 3, s=30, color=colors[group_idx], 
                    alpha=0.6, edgecolor='white', linewidth=0.5)
        
        ax.text(0, 0, f"Total: {sum(group_sizes)}\nfood items",
                ha='center', va='center',
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.9))

    ax.set_ylim(0, 15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)

    plt.title('HELIAD Food Groups Sunburst\nOuter ring: Groups | Inner dots: Individual foods', 
            fontsize=14, fontweight='bold', pad=30)
    plt.tight_layout()
    return plt

def boxplot_z_score_foods_per_multi_group(hdata):
    return boxplot_z_score_foods_per_group(hdata, nutri_dict_grouped)

def boxplot_z_score_foods_per_single_group(hdata):
    return boxplot_z_score_foods_per_group(hdata, single_items)

def boxplot_z_score_foods_per_group(hdata, group_dict):
    food_group_cols = list(group_dict.keys())
    n_cols = 4
    n_rows = (len(food_group_cols) + n_cols - 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15))
    axes = axes.flatten()
    # plt.subplots_adjust(hspace=1.6)

    for idx, (category, foods) in enumerate(group_dict.items()):
        if idx >= len(axes):
            break

        if (isinstance(foods, str)):
            foods = [foods]
            
        ax = axes[idx]
        
        category_data = []
        category_labels = []
        
        for food in foods:
            for ffq_code, food_name in nutridict.items():
                if food_name == food:
                    data = hdata[ffq_code].dropna()
                    if len(data) > 0:
                        # z_data = (data - data.mean()) / data.std()
                        category_data.append(data)
                        category_labels.append(food)
                        break
        
        if category_data:
            bp = ax.boxplot(category_data, vert=True, patch_artist=True,
                        showfliers=False, whis=[5, 95])
            
            for patch in bp['boxes']:
                patch.set_facecolor(category_colors[category])
                patch.set_alpha(0.7)
            
            ax.set_xticks(range(1, len(category_labels) + 1))
            ax.set_xticklabels(category_labels, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('Z-score')
            ax.set_title(f'{category} (n={len(category_data)} items)', 
                        fontsize=11, fontweight='bold')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)  # Reference line
            ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('HELIAD Study: Food Consumption by Category (Z-scores)', 
                fontsize=16, fontweight='bold', y=1.02)
    for i in range(len(food_group_cols), len(axes)):
        axes[i].set_visible(False)
        
    # plt.tight_layout()
    return plt